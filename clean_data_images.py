import os
import shutil
from tqdm import tqdm
from multiprocessing import Process
import glob


def func1():
    # List scenes folders:
    data_dir = "/ibex/scratch/abdelrem/scannet_dataset/scannet/scans"
    scenes_names = next(os.walk(data_dir))[1]
    for scene_name in tqdm(scenes_names):
        scene_path = os.path.join(data_dir, scene_name)
        img_folders_names = next(os.walk(scene_path))[1]
        if "images" in img_folders_names:
            shutil.rmtree(os.path.join(scene_path, "images"))
        if "images_100" in img_folders_names:
            shutil.rmtree(os.path.join(scene_path, "images_100"))
        if "images_cocoon" in img_folders_names:
            shutil.rmtree(os.path.join(scene_path, "images_cocoon"))
        if "images_cocoon_dense" in img_folders_names:
            shutil.rmtree(os.path.join(scene_path, "images_cocoon_dense"))


def func2():
    # List scenes folders:
    data_dir = "/ibex/scratch/abdelrem/scannet_dataset/scannet/scans"
    scenes_names = next(os.walk(data_dir))[1]
    for scene_name in tqdm(scenes_names):
        scene_path = os.path.join(data_dir, scene_name)
        imgs_path = os.path.join(scene_path, "images_cocoon_geo_dense")
        obj_ids = next(os.walk(imgs_path))[1]
        for obj_id in obj_ids:
            obj_path = os.path.join(imgs_path, obj_id)
            imgs = glob.glob(obj_path+"/*")
            cocoonAngles = ["_0.", "_30.", "_60.", "_-30.", "_-60."]
            for angle in cocoonAngles:
                for img in imgs:
                    if angle in img:
                        print("angle = ", angle)



if __name__ == '__main__':
    p1 = Process(target=func2)
    p1.start()
    p1.join()