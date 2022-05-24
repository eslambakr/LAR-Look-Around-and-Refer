import random
import torch
import time
import os
import numpy as np
from torch.utils.data import Dataset
from functools import partial
from .utils import dataset_to_dataloader, max_io_workers
import cv2
import math
import os
import random
import torch
from pytorch_transformers.tokenization_bert import BertTokenizer
import albumentations as A
# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import check_segmented_object_order, sample_scan_object, pad_samples, objects_bboxes, pad_images, pad_geo
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform
from ...data_generation.nr3d import decode_stimulus_string


class ListeningDataset(Dataset):
    def __init__(self, references, scans, vocab, max_seq_len, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None,
                 visualization=False, feat2dtype=None,
                 num_class_dim=525, evalmode=False, img_enc=False, load_imgs=False, mode=None, imgsize=32,
                 train_vis_enc_only=False, cocoon=False,
                 twoStreams=False, sceneCocoonPath=None, context_info_2d_cached_file="None"):

        self.references = references
        self.scans = scans
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation
        self.img_enc = img_enc
        self.load_imgs = load_imgs
        self.mode = mode
        self.imgsize = imgsize
        self.train_vis_enc_only = train_vis_enc_only
        self.cocoon = cocoon
        self.twoStreams = twoStreams
        self.sceneCocoonPath = sceneCocoonPath
        self.feat2dtype = feat2dtype
        self.max_2d_view = 5
        self.num_class_dim = num_class_dim
        self.evalmode = evalmode
        self.context_info_2d_cached_file = context_info_2d_cached_file

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')
        assert self.bert_tokenizer.encode(self.bert_tokenizer.pad_token) == [0]

        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        tokens = np.array(self.vocab.encode(ref['tokens'], self.max_seq_len), dtype=np.long)
        is_nr3d = ref['dataset'] == 'nr3d'

        return scan, target, tokens, ref['tokens'], is_nr3d

    def prepare_distractors(self, scan, target):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        distractors = [o for o in scan.three_d_objects if
                       (o.instance_label == target_label and (o != target))]

        # Then all more objects up to max-number of distractors
        already_included = {target_label}
        clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
        if self.mode == "train":
            np.random.shuffle(clutter)

        distractors.extend(clutter)
        distractors = distractors[:self.max_distractors]
        if self.mode == "train":
            np.random.shuffle(distractors)

        return distractors

    # ----------------------------------------------------------------------------------------------
    #                           Projecting 3D Point-clouds to 2D images
    # ----------------------------------------------------------------------------------------------
    def convert_world2image_cord_vectorized_ver(self, obj_pc_voxel, m, intrinsic):
        """
        This function exactly like "convert_world2image_cord" but this is the vectorized version of it.
        """
        projected_points = np.zeros_like(obj_pc_voxel)
        m = m[:3, :]
        m = np.repeat(m[np.newaxis, :, :], len(obj_pc_voxel), axis=0)  # [num_objs, 3, 4]
        intrinsic = np.repeat(intrinsic[np.newaxis, :, :], len(obj_pc_voxel), axis=0)  # [num_objs, 3, 3]
        obj_pc_voxel = np.hstack((obj_pc_voxel, np.ones((len(obj_pc_voxel), 1))))  # [num_objs, 4]
        p_cam = np.matmul(m,
                          np.expand_dims(obj_pc_voxel, axis=-1))  # [num_objs, 3, 4].[num_objs, 4, 1]=[num_objs, 3, 1]
        p_img = np.matmul(intrinsic, p_cam)  # [num_objs, 3, 3].[num_objs, 3, 1] = [num_objs, 3, 1]
        p_pixel = p_img[:, :, 0] * (1 / p_img[:, -1, :])
        return p_pixel

    def lookat(self, center, target, up):
        """
        https://github.com/isl-org/Open3D/issues/2338
        https://stackoverflow.com/questions/54897009/look-at-function-returns-a-view-matrix-with-wrong-forward-position-python-im
        https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
        https://www.youtube.com/watch?v=G6skrOtJtbM
        f: forward
        s: right
        u: up
        """
        f = (target - center)
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        u = u / np.linalg.norm(u)

        m = np.zeros((4, 4))
        m[0, :-1] = -s
        m[1, :-1] = u
        m[2, :-1] = f
        m[-1, -1] = 1.0

        t = np.matmul(-m[:3, :3], center)
        m[:3, 3] = t

        return m

    def get_perpendicular_vector_on_plane(self, plane, point):
        """
        Get perpendicular vector on plane. Use the point to determine the direction of the vector.
        The plane is represented by 4 corners [4, 3].
        """
        O = self.get_plane_center(plane)
        # O = np.array([plane[0, 0], plane[0, 1], plane[0, 2]])  # Corner to be used as the origin
        V1 = np.array([plane[1, 0], plane[1, 1], plane[2, 2]]) - O  # Relative vectors
        V2 = np.array([plane[2, 0], plane[2, 1], plane[2, 2]]) - O
        V1 = V1 / np.linalg.norm(V1)  # Normalise vectors
        V2 = V2 / np.linalg.norm(V2)
        # Take the cross product
        perp = np.cross(V1, V2)

        direction = perp / np.linalg.norm(perp)
        # To avoid looking from outside the room
        check_dir = [np.sign(point[0] - O[0]), np.sign(point[1] - O[1]), np.sign(point[2] - O[2])]
        for i in range(3):  # loop on x, y ,z
            if np.sign(direction[i]) != np.sign(check_dir[i]):
                direction[i] = direction[i] * -1
        return direction

    def calculateDistance(self, p1, p2):
        """
        Calculate distance between two points in the space/3D.
        """
        dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
        return dist

    def get_plane_center(self, plane):
        # Get center of the 4 points/polygon:
        x = [p[0] for p in plane]
        y = [p[1] for p in plane]
        z = [p[2] for p in plane]
        centroid = np.array([sum(x) / len(plane), sum(y) / len(plane), sum(z) / len(plane)])
        return centroid

    def get_plane_area(self, plane):
        for i in range(3):  # loop on x, y, z
            if len(np.unique(plane[:, i])) == 1:
                break
        plane = np.delete(plane, i, axis=1)
        x1, x2 = plane[:, 0].min(), plane[:, 0].max()
        y1, y2 = plane[:, 1].min(), plane[:, 1].max()
        area = (y2 - y1) * (x2 - x1)
        return area

    def get_nearst_face_from_point(self, point, faces):
        """
        Calculate distances between point in 3D and each face to choose the nearst face/plane.
        """
        # Get the largest two faces:
        if self.get_plane_area(faces[0]) >= self.get_plane_area(faces[2]):
            faces = faces[:2]
        else:
            faces = faces[2:]
        min_dist = self.calculateDistance(point, self.get_plane_center(faces[0]))
        nearst_face = faces[0]
        for face in faces:
            dist = self.calculateDistance(point, self.get_plane_center(face))
            if dist < min_dist:
                min_dist = dist
                nearst_face = face
        return nearst_face

    def get_obj_faces(self, obj):
        """
        Takes an object and returns a list contains the 6 faces: [x1, x2, y1, y2, z1, z2]
        """
        obj_faces = []
        obj_faces.append(obj.get_bbox().x_faces()[0])
        obj_faces.append(obj.get_bbox().x_faces()[1])
        obj_faces.append(obj.get_bbox().y_faces()[0])
        obj_faces.append(obj.get_bbox().y_faces()[1])
        obj_faces.append(obj.get_bbox().z_faces()[0])
        obj_faces.append(obj.get_bbox().z_faces()[1])
        return obj_faces

    def get3d_box_from_pcs(self, pc):
        """
        Given point-clouds that represent object or scene return the 3D dimension of the 3D box that contains the PCs.
        """
        w = pc[:, 0].max() - pc[:, 0].min()
        l = pc[:, 1].max() - pc[:, 1].min()
        h = pc[:, 2].max() - pc[:, 2].min()
        return w, l, h

    def proj_3d_to_2dimg(self, scan):
        # This ratio will be used as a percentage from the original dimension
        # to take the surrounding points of a certain object
        agument = True
        voxel_ratio = 2 / 100
        k_size = 5
        desired_shape = (32, 32)
        max_grid_dim = 1200
        distPointOptions = ["boxCenter", "boxFaceCenter"]
        distPointArg = distPointOptions[0]
        up_vector = np.array([0, 0, -1])
        pc = scan.pc
        # get scene dimensions (w, l, h):
        w, l, h = self.get3d_box_from_pcs(pc)

        # get center of the scene
        scene_center = np.array([pc[:, 0].max() - w / 2, pc[:, 1].max() - l / 2, pc[:, 2].max() - h / 2])

        for i, obj in enumerate(scan.three_d_objects):
            intrinsic = np.array([[623.53829072, 0., 359.5], [0., 623.53829072, 359.5], [0., 0., 1.]])
            # Voxelizing the obj point-clouds:
            obj_pc = pc[obj.points]
            # get scene dimensions (w, l, h):
            w, l, h = self.get3d_box_from_pcs(obj_pc)
            x_bound = [obj_pc[:, 0].min() - (w * voxel_ratio), obj_pc[:, 0].max() + (w * voxel_ratio)]
            y_bound = [obj_pc[:, 1].min() - (l * voxel_ratio), obj_pc[:, 1].max() + (l * voxel_ratio)]
            z_bound = [obj_pc[:, 2].min() - (h * voxel_ratio), obj_pc[:, 2].max() + (h * voxel_ratio)]
            # filter the voxel from the whole scene:
            filtered_idx = np.where((pc[:, 0] < x_bound[1]) & (pc[:, 0] > x_bound[0])
                                    & (pc[:, 1] < y_bound[1]) & (pc[:, 1] > y_bound[0])
                                    & (pc[:, 2] < z_bound[1]) & (pc[:, 2] > z_bound[0]))
            obj_pc_voxel = scan.pc[filtered_idx]
            obj_color_voxel = scan.color[filtered_idx]

            # Get camera pos & the target point:
            # ----------------------------------
            faces = self.get_obj_faces(obj)[:4]  # exclude z faces
            nearst_face = self.get_nearst_face_from_point(scene_center, faces)
            direction = self.get_perpendicular_vector_on_plane(plane=nearst_face, point=scene_center)
            box_center = np.array([obj.get_bbox().cx, obj.get_bbox().cy, obj.get_bbox().cz])
            if distPointArg == "boxCenter":
                O = box_center
            elif distPointArg == "boxFaceCenter":
                O = self.get_plane_center(nearst_face)
            else:
                O = None

            # set the camera away from the object at certain distance (d)
            # https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
            if agument:
                d = np.random.uniform(1.5, 4)
                up_d = np.random.uniform(0.5, 2.5)
            else:
                d = 2
                up_d = 1
            camera_pos = O + (d * direction)
            camera_pos[-1] = camera_pos[-1] + up_d  # lift the camera

            m = self.lookat(camera_pos, O, up_vector)
            projected_points = self.convert_world2image_cord_vectorized_ver(obj_pc_voxel, m, intrinsic)

            # Shift -ve points:
            projected_points[:, 0] = projected_points[:, 0] - projected_points[:, 0].min()
            projected_points[:, 1] = projected_points[:, 1] - projected_points[:, 1].min()

            ptXYZRGB = np.hstack((projected_points, obj_color_voxel))

            # Create the grid:
            grid = np.ones((min(math.ceil(ptXYZRGB[:, 1].max()) + k_size, max_grid_dim + k_size),
                            min(math.ceil(ptXYZRGB[:, 0].max()) + k_size, max_grid_dim + k_size), 3)) * 255
            # check grid boundaries:
            if math.ceil(ptXYZRGB[:, 1].max()) > max_grid_dim:
                ptXYZRGB[:, 1] = (ptXYZRGB[:, 1] / ptXYZRGB[:, 1].max()) * max_grid_dim
            if math.ceil(ptXYZRGB[:, 0].max()) > max_grid_dim:
                ptXYZRGB[:, 0] = (ptXYZRGB[:, 0] / ptXYZRGB[:, 0].max()) * max_grid_dim

            # Interpolate each pixel:
            for j1 in range(k_size):
                for j2 in range(k_size):
                    grid[(ptXYZRGB[:, 1] + j1).astype(int),
                         (ptXYZRGB[:, 0] + j2).astype(int)] = ptXYZRGB[:, :-4:-1] * 255

            # Downsize the array:  # [H, W]
            grid = cv2.resize(grid, desired_shape)
            scan.three_d_objects[i].set_2d_img(grid)

        return scan

    def load_projected_2dimg(self, scan):
        desired_shape = self.imgsize
        for i, obj in enumerate(scan.three_d_objects):
            obj_pth = obj.imgsPath
            if self.mode == "train":
                augment = True
                train_transform = None
                # Augmentation:
                if augment:
                    train_transform = A.Compose(
                        [
                            A.OneOf([A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                                     A.VerticalFlip(p=0.5),
                                     A.HorizontalFlip(p=0.5)]),
                            A.OneOf([A.RandomCrop(height=int(self.imgsize * 0.95), width=int(self.imgsize * 0.95)),
                                     A.InvertImg(p=0.1)]),
                            A.OneOf([A.OpticalDistortion(distort_limit=0.2, p=0.5),
                                     A.ChannelShuffle(p=0.5),
                                     A.MotionBlur(p=0.5),
                                     A.GlassBlur(p=0.5)]),
                            A.OneOf([A.Emboss(p=0.5),
                                     A.Sharpen(p=0.5),
                                     A.RandomGridShuffle(p=0.1),
                                     A.RandomGamma(p=0.2)]),
                        ]
                    )
                img_id = random.randint(0, 49)
                # img_id = 0
                if self.cocoon:
                    cocoonAngles = [0, 30, 60, -30, -60]
                    img = []
                    for angle in cocoonAngles:
                        imgName = os.path.join(obj_pth, str(img_id)) + "_" + str(angle) + ".jpg"
                        imgTemp = cv2.imread(imgName)
                        imgTemp = cv2.resize(imgTemp, (desired_shape, desired_shape))
                        if train_transform is not None:
                            imgTemp = train_transform(image=imgTemp)["image"]
                            imgTemp = cv2.resize(imgTemp, (desired_shape, desired_shape))

                        imgTemp = imgTemp.astype(float) / 255
                        img.append(imgTemp)
                else:
                    angle = 0
                    imgName = os.path.join(obj_pth, str(img_id)) + "_" + str(angle) + ".jpg"
                    img = cv2.imread(imgName)
                    img = cv2.resize(img, (desired_shape, desired_shape))
                    if train_transform is not None:
                        img = train_transform(image=img)["image"]
                        img = cv2.resize(img, (desired_shape, desired_shape))

                    img = img.astype(float) / 255
            else:
                img_id = 100
                if self.cocoon:
                    cocoonAngles = [0, 30, 60, -30, -60]
                    img = []
                    for angle in cocoonAngles:
                        imgName = os.path.join(obj_pth, str(img_id)) + "_" + str(angle) + ".jpg"
                        imgTemp = cv2.imread(imgName)
                        imgTemp = cv2.resize(imgTemp, (desired_shape, desired_shape))
                        imgTemp = imgTemp.astype(float) / 255
                        img.append(imgTemp)
                else:
                    angle = 0
                    imgName = os.path.join(obj_pth, str(img_id)) + "_" + str(angle) + ".jpg"
                    img = cv2.imread(imgName)
                    img = cv2.resize(img, (desired_shape, desired_shape))
                    img = img.astype(float) / 255

            scan.three_d_objects[i].set_2d_img(img)

        return scan

    def load_2dgeo_info(self, scan):
        for i, obj in enumerate(scan.three_d_objects):
            obj_pth = obj.imgsPath
            if self.mode == "train":
                img_id = 0
            else:
                img_id = 100
            if self.cocoon:
                cocoonAngles = [0, 30, 60, -30, -60]
                geoInfo = []
                for angle in cocoonAngles:
                    # img_id = random.randint(0, 99)

                    npGeoFilePath = os.path.join(obj_pth, str(img_id)) + "_" + str(angle) + ".npy"
                    geo_info = np.load(npGeoFilePath)
                    geoInfo.append(geo_info)
            else:
                # img_id = random.randint(1, 99)
                img_id = 0
                angle = 0
                npGeoFilePath = os.path.join(obj_pth, str(img_id)) + "_" + str(angle) + ".npy"
                geoInfo = np.load(npGeoFilePath)

            scan.three_d_objects[i].set_2d_geo_info(geoInfo)
        return scan

    def load_projected_2dsceneimgs(self, scan):
        img = []
        for i in range(5):
            desired_shape = 128
            imgName = os.path.join(self.sceneCocoonPath, scan.scan_id) + "_" + str(i) + ".jpg"
            imgTemp = cv2.imread(imgName)
            imgTemp = cv2.resize(imgTemp, (desired_shape, desired_shape))
            imgTemp = imgTemp.astype(float) / 255
            img.append(imgTemp)
        scan.set_2d_sceneimgs(img)
        return scan

    def __getitem__(self, index):

        res = dict()
        scan, target, tokens, text_tokens, is_nr3d = self.get_reference_data(index)

        # BERT tokenize
        token_inds = torch.zeros(self.max_seq_len, dtype=torch.long)
        indices = self.bert_tokenizer.encode(
            ' '.join(text_tokens), add_special_tokens=True)
        indices = indices[:self.max_seq_len]
        token_inds[:len(indices)] = torch.tensor(indices)
        token_num = torch.tensor(len(indices), dtype=torch.long)

        # Generate 2d image projected from 3d point clouds
        if self.img_enc and not self.load_imgs:
            scan = self.proj_3d_to_2dimg(scan)
        elif self.img_enc and self.load_imgs:
            scan = self.load_projected_2dimg(scan)

        # Load Geo info:
        scan = self.load_2dgeo_info(scan)

        # Load 2d images projected from 3d point clouds representing the whole scene:
        if self.sceneCocoonPath:
            scan = self.load_projected_2dsceneimgs(scan)
            res['sceneImgs'] = np.array(scan.sceneimgs)

        # Make a context of distractors
        context = self.prepare_distractors(scan, target)

        # Add target object in 'context' list
        target_pos = np.random.randint(len(context) + 1)
        context.insert(target_pos, target)

        if self.load_imgs and not self.twoStreams:
            res['context_size'] = len(context)
            # res['objects'] = None
        else:
            # sample point/color for them
            samples = np.array([sample_scan_object(o, self.points_per_object) for o in context])
            if self.object_transformation is not None:
                samples, offset = self.object_transformation(samples)
                res['obj_offset'] = np.zeros((self.max_context_size, offset.shape[1])).astype(np.float32)
                res['obj_offset'][:len(offset), :] = offset.astype(np.float32)

            res['context_size'] = len(samples)
            # take care of padding, so that a batch has same number of N-objects across scans.
            res['objects'] = pad_samples(samples, self.max_context_size)  # [max_context_size, 1024, 6]

        # get object's images
        if self.img_enc:
            objs_img = np.array([o.img for o in context])  # [num_obj, img_h, img_w, 3]
            res['objectsImgs'] = pad_images(objs_img, self.max_context_size)  # [max_context_size, img_h, img_w, 3]

        # get Geo Info:
        objs_geo = np.array([o.geo_info for o in context])  # [num_obj, 30]
        res['objectsGeo'] = pad_geo(objs_geo, self.max_context_size)  # [max_context_size, 30]

        # mark their classes
        res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)
        res['clsvec'] = np.zeros((self.max_context_size, self.num_class_dim)).astype(np.float32)
        for ii in range(len(res['class_labels'])):
            res['clsvec'][ii, res['class_labels'][ii]] = 1.

        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_class_mask[:len(context)] = [target.instance_label == o.instance_label for o in context]

        res['target_class'] = self.class_to_idx[target.instance_label]
        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask
        res['tokens'] = tokens
        res['token_inds'] = token_inds.numpy().astype(np.int64)
        res['token_num'] = token_num.numpy().astype(np.int64)
        res['is_nr3d'] = is_nr3d

        if self.visualization:
            distrators_pos = np.zeros((6))  # 6 is the maximum context size we used in dataset collection
            object_ids = np.zeros((self.max_context_size))
            j = 0
            for k, o in enumerate(context):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    distrators_pos[j] = k
                    j += 1
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res['utterance'] = self.references.loc[index]['utterance']
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['distrators_pos'] = distrators_pos
            res['object_ids'] = object_ids
            res['target_object_id'] = target.object_id

        # load cached 2D context information
        if self.context_info_2d_cached_file is not None and os.path.isfile(
                self.context_info_2d_cached_file + '/scannet_frames_25k_gtobjfeat_aggregate/%s.npy' % scan.scan_id):
            context_2d = np.load(
                self.context_info_2d_cached_file + '/scannet_frames_25k_gtobjfeat_aggregate/%s.npy' % scan.scan_id,
                allow_pickle=True, encoding='latin1')
            objfeat_2d = context_2d.item()['obj_feat']
            bbox_2d = context_2d.item()['obj_coord']
            bboxsize_2d = context_2d.item()['obj_size']
            obj_depth = context_2d.item()['obj_depth']
            campose_2d = context_2d.item()['camera_pose']
            ins_id_2d = context_2d.item()['instance_id']
            if (self.feat2dtype.replace('3D', '')) == 'ROI':
                featdim = 2048
            elif (self.feat2dtype.replace('3D', '')) == 'clsvec':
                featdim = self.num_class_dim
            elif (self.feat2dtype.replace('3D', '')) == 'clsvecROI':
                featdim = 2048 + self.num_class_dim
            feat_2d = np.zeros((self.max_context_size, featdim)).astype(np.float32)
            coords_2d = np.zeros((self.max_context_size, 4 + 12)).astype(np.float32)

            selected_2d_idx = 0
            selected_context_id = [o.object_id + 1 for o in context]  ## backbround included in cache, so +1
            ## only for creating tensor of the correct size
            selected_objfeat_2d = objfeat_2d[selected_context_id, selected_2d_idx, :]
            selected_bbox_2d = bbox_2d[selected_context_id, selected_2d_idx, :]
            selected_bboxsize_2d = bboxsize_2d[selected_context_id, selected_2d_idx]
            selected_obj_depth = obj_depth[selected_context_id, selected_2d_idx]
            selected_campose_2d = campose_2d[selected_context_id, selected_2d_idx, :]
            selected_ins_id_2d = ins_id_2d[selected_context_id, selected_2d_idx]
            ## Fill in randomly selected view of 2D features
            for ii in range(len(selected_context_id)):
                cxt_id = selected_context_id[ii]
                view_id = random.randint(0, max(0, int((ins_id_2d[cxt_id, :] != 0).astype(np.float32).sum()) - 1))
                selected_objfeat_2d[ii, :] = objfeat_2d[cxt_id, view_id, :]
                selected_bbox_2d[ii, :] = bbox_2d[cxt_id, view_id, :]
                selected_bboxsize_2d[ii] = bboxsize_2d[cxt_id, view_id]
                selected_obj_depth[ii] = obj_depth[cxt_id, view_id]
                selected_campose_2d[ii, :] = campose_2d[cxt_id, view_id, :]

            if self.feat2dtype != 'clsvec':
                feat_2d[:len(selected_context_id), :2048] = selected_objfeat_2d
            for ii in range(len(res['class_labels'])):
                if self.feat2dtype == 'clsvec':
                    feat_2d[ii, res['class_labels'][ii]] = 1.
                if self.feat2dtype == 'clsvecROI':
                    feat_2d[ii, 2048 + res['class_labels'][ii]] = 1.
            coords_2d[:len(selected_context_id), :] = np.concatenate(
                [selected_bbox_2d, selected_campose_2d[:, :12]], axis=-1)
            coords_2d[:, 0], coords_2d[:, 2] = coords_2d[:, 0] / 1296., coords_2d[:,
                                                                        2] / 1296.  ## norm by image size
            coords_2d[:, 1], coords_2d[:, 3] = coords_2d[:, 1] / 968., coords_2d[:, 3] / 968.
            res['feat_2d'] = feat_2d
            res['coords_2d'] = coords_2d
        return res


def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb, seed=None, gen=None):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    samplers = dict()
    is_train = referit_data['is_train']

    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=args.unit_sphere_norm, inplace=False)
    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        # if split == test remove the utterances of unique targets
        if split == 'test' and (not args.train_scanRefer):
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            print("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

            assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        dataset = ListeningDataset(references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=args.mode == 'evaluate',
                                   feat2dtype=args.feat2d,
                                   num_class_dim=525 if '00' in args.scannet_file else 608,
                                   img_enc=args.img_encoder,
                                   load_imgs=args.load_imgs,
                                   mode=split,
                                   imgsize=args.imgsize,
                                   evalmode=(args.mode == 'evaluate'),
                                   train_vis_enc_only=args.train_vis_enc_only,
                                   cocoon=args.cocoon,
                                   twoStreams=args.twoStreams,
                                   sceneCocoonPath=args.sceneCocoonPath,
                                   context_info_2d_cached_file=args.context_info_2d_cached_file)

        seed = None
        if split == 'test':
            seed = args.random_seed

        if args.distributed and split == 'train':  # E: Shouldn't distribute the data while testing
            samplers[split] = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            samplers[split] = None
        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, seed=seed,
                                                    sampler=samplers[split], gen=gen)

    return data_loaders, samplers
