import numpy as np
from shapely.geometry import Polygon, LineString
from sklearn.neighbors import NearestNeighbors
import math
import cv2
import os
import os.path
from os import path
import copy
from .cuboid import OrientedCuboid
from ..utils.plotting import plot_pointcloud


def get_angle_between_2vectors(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product) * (180 / np.pi)

    return angle


def get_point_on_circle(point, circelCenter, angle):
    """
    get a point on the circle surface given the angle.
    https://stackoverflow.com/questions/58501322/how-to-calculate-point-on-circle-from-angle-between-middle-and-other-point-on-th
    """
    x1 = circelCenter[0] + (point[0] - circelCenter[0]) * math.cos(angle) - (point[1] - circelCenter[1]) * math.sin(angle)
    y1 = circelCenter[1] + (point[0] - circelCenter[0]) * math.sin(angle) + (point[1] - circelCenter[1]) * math.cos(angle)
    return [x1, y1]


def get3d_box_from_pcs(pc):
    """
    Given point-clouds that represent object or scene return the 3D dimension of the 3D box that contains the PCs.
    """
    w = pc[:, 0].max() - pc[:, 0].min()
    l = pc[:, 1].max() - pc[:, 1].min()
    h = pc[:, 2].max() - pc[:, 2].min()
    return w, l, h


def get_plane_center(plane):
    # Get center of the 4 points/polygon:
    x = [p[0] for p in plane]
    y = [p[1] for p in plane]
    z = [p[2] for p in plane]
    centroid = np.array([sum(x) / len(plane), sum(y) / len(plane), sum(z) / len(plane)])
    return centroid


def lookat(center, target, up):
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


def convert_world2image_cord(extrinsic, intrinsic, p_world):
    """
    extrinsic: [4, 4]
    intrinsic: [3, 3]
    p_world: [3,]
    https://towardsdatascience.com/camera-calibration-fda5beb373c3
    """
    p_world = np.hstack((p_world, 1))  # [3, 1] --> [4,1]
    """
    projection_matrix = np.matmul(intrinsic, extrinsic[:3, :])  # [3, 4]
    p_img = np.matmul(projection_matrix, p_world)  # [3,]
    """
    p_cam = np.dot(extrinsic[:3, :], p_world)  # [3,4].[4,1] = [3, 1]
    p_img = np.dot(intrinsic, p_cam)  # [3, 3].[3, 1] = [3, 1]
    p_pixel = p_img*(1/p_img[-1])
    return p_pixel


def convert_world2image_cord_vectorized_ver(obj_pc_voxel, m, intrinsic):
    """
    This function exactly like "convert_world2image_cord" but this is the vectorized version of it.
    """
    projected_points = np.zeros_like(obj_pc_voxel)
    m = m[:3, :]
    m = np.repeat(m[np.newaxis, :, :], len(obj_pc_voxel), axis=0)  # [num_objs, 3, 4]
    intrinsic = np.repeat(intrinsic[np.newaxis, :, :], len(obj_pc_voxel), axis=0)  # [num_objs, 3, 3]
    obj_pc_voxel = np.hstack((obj_pc_voxel, np.ones((len(obj_pc_voxel), 1))))  # [num_objs, 4]
    p_cam = np.matmul(m, np.expand_dims(obj_pc_voxel, axis=-1))  # [num_objs, 3, 4].[num_objs, 4, 1]=[num_objs, 3, 1]
    p_img = np.matmul(intrinsic, p_cam)  # [num_objs, 3, 3].[num_objs, 3, 1] = [num_objs, 3, 1]
    p_pixel = p_img[:, :, 0] * (1 / p_img[:, -1, :])
    return p_pixel


def get_obj_faces(obj):
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


def calculateDistance(p1, p2):
    """
    Calculate distance between two points in the space/3D.
    """
    dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    return dist


def get_plane_area(plane):
    for i in range(3):  # loop on x, y, z
        if len(np.unique(plane[:, i])) == 1:
            break
    plane = np.delete(plane, i, axis=1)
    x1, x2 = plane[:, 0].min(), plane[:, 0].max()
    y1, y2 = plane[:, 1].min(), plane[:, 1].max()
    area = (y2-y1)*(x2-x1)
    return area


def get_nearst_face_from_point(point, faces, z=False):
    """
    Calculate distances between point in 3D and each face to choose the nearst face/plane.
    """
    if z == False:
        # Get the largest two faces:
        if get_plane_area(faces[0]) >= get_plane_area(faces[2]):
            faces = faces[:2]
        else:
            faces = faces[2:]
    min_dist = calculateDistance(point, get_plane_center(faces[0]))
    nearst_face = faces[0]
    for face in faces:
        dist = calculateDistance(point, get_plane_center(face))
        if dist < min_dist:
            min_dist = dist
            nearst_face = face
    return nearst_face


def get_perpendicular_vector_on_plane(plane, point):
    """
    Get perpendicular vector on plane. Use the point to determine the direction of the vector.
    The plane is represented by 4 corners [4, 3].
    """
    O = get_plane_center(plane)
    #O = np.array([plane[0, 0], plane[0, 1], plane[0, 2]])  # Corner to be used as the origin
    V1 = np.array([plane[1, 0], plane[1, 1], plane[2, 2]]) - O  # Relative vectors
    V2 = np.array([plane[2, 0], plane[2, 1], plane[2, 2]]) - O
    V1 = V1 / np.linalg.norm(V1)  # Normalise vectors
    V2 = V2 / np.linalg.norm(V2)
    # Take the cross product
    perp = np.cross(V1, V2)

    direction = perp / np.linalg.norm(perp)
    # To avoid looking from outside the room
    check_dir = [np.sign(point[0]-O[0]), np.sign(point[1]-O[1]), np.sign(point[2]-O[2])]
    for i in range(3):  # loop on x, y ,z
        if np.sign(direction[i]) != np.sign(check_dir[i]):
            direction[i] = direction[i] * -1
    return direction


def project_pc_2_img(scan, obj, saving_pth, augment=True, cocoonAngles = [0]):
    # configurations:
    voxel_ratio_org = [2/100, 2/100, 2/100]
    k_size = 5
    desired_shape = (scan.img_size, scan.img_size)
    max_grid_dim = 1200
    distPointOptions = ["boxCenter", "boxFaceCenter"]
    distPointArg = distPointOptions[0]
    up_vector = np.array([0, 0, -1])

    # get scene dimensions (w, l, h):
    w, l, h = get3d_box_from_pcs(scan.pc)

    # get center of the scene
    scene_center = np.array([scan.pc[:, 0].max() - w / 2, scan.pc[:, 1].max() - l / 2, scan.pc[:, 2].max() - h / 2])

    # Get camera pos & the target point:
    # ----------------------------------
    if obj.instance_label == "ceiling" or obj.instance_label == "floor":
        faces = get_obj_faces(obj)[4:]  # include z faces only
        nearst_face = get_nearst_face_from_point(scene_center, faces, z=True)
    else:
        faces = get_obj_faces(obj)[:4]  # exclude z faces
        nearst_face = get_nearst_face_from_point(scene_center, faces)
    direction = get_perpendicular_vector_on_plane(plane=nearst_face, point=scene_center)
    box_center = np.array([obj.get_bbox().cx, obj.get_bbox().cy, obj.get_bbox().cz])
    if distPointArg == "boxCenter":
        O = box_center
    elif distPointArg == "boxFaceCenter":
        O = get_plane_center(nearst_face)
    else:
        O = None

    intrinsic = np.array([[623.53829072, 0., 359.5], [0., 623.53829072, 359.5], [0., 0., 1.]])
    # Voxelizing the obj point-clouds:
    obj_idx = obj.points
    obj_pc = scan.pc[obj_idx]
    obj_color = scan.color[obj_idx]
    voxel_ratio = copy.deepcopy(voxel_ratio_org)
    if obj.instance_label != "ceiling" and obj.instance_label != "floor":
        angle = get_angle_between_2vectors(vector_1=abs(direction[:2]), vector_2=[1, 0])
        voxel_ratio[0] *= abs(angle / 90)
        voxel_ratio[0] = max(voxel_ratio[0], 2 / 100)
        voxel_ratio[1] *= abs((angle / 90) - 1)
        voxel_ratio[1] = max(voxel_ratio[1], 2 / 100)
    # get scene dimensions (w, l, h):
    w, l, h = get3d_box_from_pcs(obj_pc)
    x_bound = [obj_pc[:, 0].min() - (w * voxel_ratio[0]), obj_pc[:, 0].max() + (w * voxel_ratio[0])]
    y_bound = [obj_pc[:, 1].min() - (l * voxel_ratio[1]), obj_pc[:, 1].max() + (l * voxel_ratio[1])]
    z_bound = [obj_pc[:, 2].min() - (h * voxel_ratio[2]), obj_pc[:, 2].max() + (h * voxel_ratio[2])]
    # filter the voxel from the whole scene:
    filtered_idx = np.where((scan.pc[:, 0] < x_bound[1]) & (scan.pc[:, 0] > x_bound[0])
                            & (scan.pc[:, 1] < y_bound[1]) & (scan.pc[:, 1] > y_bound[0])
                            & (scan.pc[:, 2] < z_bound[1]) & (scan.pc[:, 2] > z_bound[0]))
    obj_pc_voxel = scan.pc[filtered_idx]
    obj_color_voxel = scan.color[filtered_idx]

    # set the camera away from the object at certain distance (d)
    # https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
    if augment:
        d = np.random.uniform(1.5, 4)
        up_d = np.random.uniform(0.5, 2.5)
        dir_x = np.random.uniform(0.01, 0.2)
        dir_y = np.random.uniform(0.01, 0.2)
        direction[0] += dir_x
        direction[1] += dir_y
    else:
        d = 2
        up_d = 1
        if obj.instance_label == "ceiling" or obj.instance_label == "floor":
            direction[1] += 0.1
    camera_pos = O + (d * direction)

    # Take cocoon shots for the object: (Photo session :D)
    org_camera_pos = copy.deepcopy(camera_pos)
    for angle in cocoonAngles:
        if augment:
            added_angle = np.random.uniform(5, 25)
            added_angle += angle
        else:
            added_angle = angle
        camera_pos[:2] = get_point_on_circle(org_camera_pos[:2], O[:2], angle=added_angle * np.pi / 180)
        camera_pos[-1] = org_camera_pos[-1] + up_d  # lift the camera

        m = lookat(camera_pos, O, up_vector)

        projected_points = convert_world2image_cord_vectorized_ver(obj_pc_voxel, m, intrinsic)
        camProjected_points = copy.deepcopy(projected_points)

        # Shift -ve points:
        projected_points[:, 0] = projected_points[:, 0] - projected_points[:, 0].min()
        projected_points[:, 1] = projected_points[:, 1] - projected_points[:, 1].min()

        ptXYZRGB = np.hstack((projected_points, obj_color_voxel))
        ptXYZRGB_copy = copy.deepcopy(ptXYZRGB)

        # Create the grid:
        # TODO: Eslam should investigate into this issue (happens at scene0329_02)
        if math.isinf(ptXYZRGB[:, 1].max()) or math.isinf(ptXYZRGB[:, 0].max()):
            print("Inf Error caused because scene: ", scan.scan_id)
            print("Inf Error caused because object: ", obj.instance_label)
            grid = np.ones((500, 500)) * 255
            # Downsize the array:  # [H, W]
            grid = cv2.resize(grid, desired_shape)
            cv2.imwrite(saving_pth + "_" + str(angle) + ".jpg", grid)

            # TODO: E add condition here on args.geo
            # Add Geometry info for each 2d image in world space:
            objGeoInfo = np.concatenate((np.array(obj.get_bbox().corners).flatten(), camera_pos, direction))
            np.save(saving_pth + "_" + str(angle), objGeoInfo)
            continue
        grid = np.ones((min(math.ceil(ptXYZRGB[:, 1].max()) + k_size, max_grid_dim + k_size),
                        min(math.ceil(ptXYZRGB[:, 0].max()) + k_size, max_grid_dim + k_size), 3)) * 255
        # check grid boundaries:
        if math.ceil(ptXYZRGB[:, 1].max()) > max_grid_dim:
            ptXYZRGB[:, 1] = (ptXYZRGB[:, 1] / ptXYZRGB[:, 1].max()) * max_grid_dim
        if math.ceil(ptXYZRGB[:, 0].max()) > max_grid_dim:
            ptXYZRGB[:, 0] = (ptXYZRGB[:, 0] / ptXYZRGB[:, 0].max()) * max_grid_dim

        # Overlap the original object over the rest of the scene:
        projected_obj_points = convert_world2image_cord_vectorized_ver(obj_pc, m, intrinsic)
        projected_obj_points[:, 0] = projected_obj_points[:, 0] - camProjected_points[:, 0].min()
        projected_obj_points[:, 1] = projected_obj_points[:, 1] - camProjected_points[:, 1].min()
        objptXYZRGB = np.hstack((projected_obj_points, obj_color))
        if math.ceil(ptXYZRGB_copy[:, 1].max()) > max_grid_dim:
            objptXYZRGB[:, 1] = (objptXYZRGB[:, 1] / ptXYZRGB_copy[:, 1].max()) * max_grid_dim
        if math.ceil(ptXYZRGB_copy[:, 0].max()) > max_grid_dim:
            objptXYZRGB[:, 0] = (objptXYZRGB[:, 0] / ptXYZRGB_copy[:, 0].max()) * max_grid_dim

        # Interpolate each pixel:
        for j1 in range(k_size):
            for j2 in range(k_size):
                grid[(ptXYZRGB[:, 1] + j1).astype(int),
                     (ptXYZRGB[:, 0] + j2).astype(int)] = ptXYZRGB[:, :-4:-1] * 255
        else:
            grid[(ptXYZRGB[:, 1]).astype(int),
                 (ptXYZRGB[:, 0]).astype(int)] = ptXYZRGB[:, :-4:-1] * 255

        # Downsize the array:  # [H, W]
        grid = cv2.resize(grid, desired_shape)
        cv2.imwrite(saving_pth+"_"+str(angle)+".jpg", grid)

        # TODO: E add condition here on args.geo
        # Add Geometry info for each 2d image in world space:
        objGeoInfo = np.concatenate((np.array(obj.get_bbox().corners).flatten(), camera_pos, direction))
        np.save(saving_pth+"_"+str(angle), objGeoInfo)


class ThreeDObject(object):
    """
    Representing a ScanNet 3D Object
    rot=np.eye(N=3)
    """

    def __init__(self, scan, object_id, points, instance_label, rot=0):
        self.rot = rot
        self.scan = scan
        self.object_id = object_id
        self.points = points
        self.instance_label = instance_label

        self.axis_aligned_bbox = None
        self.is_axis_aligned_bbox_set = False

        self.object_aligned_bbox = None
        self.has_object_aligned_bbox = False

        self.front_direction = None
        self.has_front_direction = False
        self._use_true_instance = True

        self.pc = None  # The point cloud (xyz)
        self.normalized_pc = None  # The normalized point cloud (xyz) in unit sphere
        self.color = None  # The point cloud (RGB) values
        self.img = None
        if scan.save_jpg:
            scansDataRoot = scan.top_scan_dir
            numImgPerObj = self.scan.camaug
            if scan.cocoon:
                cocoonAngles = [0, 30, 60, -30, -60]
                folderName = "images_cocoon_geo"
                if scan.load_dense:
                    folderName += "_dense"
                img_pth = os.path.join(scansDataRoot, scan.scan_id, folderName)
            else:
                cocoonAngles = [0]
                img_pth = os.path.join(scansDataRoot, scan.scan_id, "images_100")

            # create images folder:
            isExist = os.path.exists(img_pth)
            if not isExist:
                os.makedirs(img_pth)
            obj_pth = os.path.join(img_pth, str(self.object_id))
            isExist = os.path.exists(obj_pth)
            if not isExist:
                os.makedirs(obj_pth)
            # Project PCs to N augmented images and save them:
            self.imgsPath = obj_pth

            # TODO: Eslam, I will clean this:
            # Eslam: To generate Nr3d data add "or True" to generate data for "_00" files:
            if not "_00/" in os.path.join(obj_pth, str(0))+"_"+str(0)+".jpg" or True:
                # """
                project_pc_2_img(scan, obj=self, saving_pth=os.path.join(obj_pth, str(0)), augment=False,
                                 cocoonAngles=cocoonAngles)
                # """
                """
                start = 100
                for i in range(start+1, start+numImgPerObj+1):
                    imgName = os.path.join(obj_pth, str(i))
                    project_pc_2_img(scan, obj=self, saving_pth=imgName, augment=True, cocoonAngles=cocoonAngles)
                """
                # """
                project_pc_2_img(scan, obj=self, saving_pth=os.path.join(obj_pth, str(100)), augment=False,
                                 cocoonAngles=cocoonAngles)
                # """

            # Eslam: Instead of storing the whole points for each object,
            # store only sub-sample of it to make pkl smaller
            n_samples, n_points = 1024, len(self.points)
            idx = np.random.choice(n_points, n_samples, replace=n_points < n_samples)
            self.pc = self.scan.pc[self.points][idx]
            self.color = self.scan.color[self.points][idx]
            self.scan = None
            self.points = None

    def set_2d_img(self, img):
        self.img = img

    def set_2d_geo_info(self, geo_info):
        self.geo_info = geo_info

    @property
    def instance_label(self):
        if self._use_true_instance:
            return self._instance_label
        else:
            return self.semantic_label()

    @instance_label.setter
    def instance_label(self, instance_label):
        self._instance_label = instance_label

    def plot(self, with_color=True):
        pc = self.get_pc()
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        color = None
        if with_color:
            color = self.color
        return plot_pointcloud(x, y, z, color=color)

    def z_min(self):
        bbox = self.get_axis_align_bbox()
        return bbox.extrema[2]

    def z_max(self):
        bbox = self.get_axis_align_bbox()
        return bbox.extrema[5]

    def set_axis_align_bbox(self):
        pc = self.get_pc()

        cx, cy, cz = (np.max(pc, axis=0) + np.min(pc, axis=0)) / 2.0
        lx, ly, lz = np.max(pc, axis=0) - np.min(pc, axis=0)
        assert (lx > 0 and ly > 0 and lz > 0)

        self.axis_aligned_bbox = OrientedCuboid(cx, cy, cz, lx, ly, lz, self.rot)
        self.is_axis_aligned_bbox_set = True

    def get_axis_align_bbox(self):
        if self.is_axis_aligned_bbox_set:
            pass
        else:
            self.set_axis_align_bbox()
        return self.axis_aligned_bbox

    def normalize_pc(self):
        """
        Normalize the object's point cloud to a unit sphere centered at the origin point
        """
        assert (self.pc is not None)
        point_set = self.pc - np.expand_dims(np.mean(self.pc, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        self.normalized_pc = point_set / dist  # scale

    def set_pc(self, normalize=False):
        if self.pc is None:
            self.pc = self.scan.pc[self.points]

        if normalize and self.normalized_pc is None:
            self.normalize_pc()

        if self.color is None:
            self.color = self.scan.color[self.points]

    def get_pc(self, normalized=False):
        # Set the pc if not previously initialized
        self.set_pc(normalized)

        if normalized:
            return self.normalized_pc

        return self.pc

    def set_object_aligned_bbox(self, cx, cy, cz, lx, ly, lz, rot):
        self.object_aligned_bbox = OrientedCuboid(cx, cy, cz, lx, ly, lz, rot)
        self.has_object_aligned_bbox = True

    def get_bbox(self, axis_aligned=False):
        """if you have object-align return this, else compute/return axis-aligned"""
        if not axis_aligned and self.has_object_aligned_bbox:
            return self.object_aligned_bbox
        else:
            return self.get_axis_align_bbox()

    def iou_2d(self, other):
        a = self.get_bbox(axis_aligned=True).corners
        b = other.get_bbox(axis_aligned=True).corners

        a_xmin, a_xmax = np.min(a[:, 0]), np.max(a[:, 0])
        a_ymin, a_ymax = np.min(a[:, 1]), np.max(a[:, 1])

        b_xmin, b_xmax = np.min(b[:, 0]), np.max(b[:, 0])
        b_ymin, b_ymax = np.min(b[:, 1]), np.max(b[:, 1])

        box_a = [a_xmin, a_ymin, a_xmax, a_ymax]
        box_b = [b_xmin, b_ymin, b_xmax, b_ymax]

        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])

        # compute the area of intersection rectangle
        inter_area = max(0, xB - xA) * max(0, yB - yA)

        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        iou = inter_area / float(box_a_area + box_b_area - inter_area)
        i_ratios = [inter_area / float(box_a_area), inter_area / float(box_b_area)]
        a_ratios = [box_a_area / box_b_area, box_b_area / box_a_area]

        return iou, i_ratios, a_ratios

    def visualize_axis_align_bbox(self, axis=None):
        bbox = self.get_axis_align_bbox()
        return bbox.plot(axis=axis)

    def color(self):
        return self.scan.color[self.points]

    def intersection(self, other, axis=2):
        bbox = self.get_bbox(axis_aligned=True)
        l_min, l_max = bbox.extrema[axis], bbox.extrema[axis + 3]

        other_bbox = other.get_bbox(axis_aligned=True)
        other_l_min, other_l_max = other_bbox.extrema[axis], other_bbox.extrema[axis + 3]

        a = max(l_min, other_l_min)
        b = min(l_max, other_l_max)
        i = b - a

        return i, i / (l_max - l_min), i / (other_l_max - other_l_min)

    def semantic_label(self):
        one_point = self.scan.semantic_label[self.points[0]]
        return self.scan.dataset.idx_to_semantic_cls(one_point)

    def distance_from_other_object(self, other, optimized=False):
        if optimized:
            z_face = self.get_bbox().z_faces()[0]  # Top face
            points = tuple(map(tuple, z_face[:, :2]))  # x, y coordinates
            center = (self.get_bbox().cx, self.get_bbox().cy)

            other_z_face = other.get_bbox().z_faces()[0]
            other_points = tuple(map(tuple, other_z_face[:, :2]))
            other_center = (other.get_bbox().cx, other.get_bbox().cy)

            cent_line = LineString([center, other_center])
            return cent_line.intersection(Polygon(points + other_points).convex_hull).length
        else:
            nn = NearestNeighbors(n_neighbors=1).fit(self.get_pc())
            distances, _ = nn.kneighbors(other.get_pc())
            res = np.min(distances)
        return res

    def sample(self, n_samples, normalized_pc=False):
        """sub-sample its pointcloud and color"""
        xyz = self.get_pc(normalized=normalized_pc)
        color = self.color

        #n_points = len(self.points)
        #assert xyz.shape[0] == len(self.points)
        n_points = len(xyz)
        assert xyz.shape[0] == n_points

        # Up-sample or Down-samples points in the object to fix number of points that represents each object:

        idx = np.random.choice(n_points, n_samples, replace=n_points < n_samples)

        return {
            'xyz': xyz[idx],
            'color': color[idx],
        }
