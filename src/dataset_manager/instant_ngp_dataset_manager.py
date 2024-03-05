import os
import glob
import shutil
import subprocess

from dataset_manager.json_manager import JsonDatasetManager
from dataset_manager.abstract_dataset_manager import AbstractDatasetManager


class InstantNGPDatasetManager(AbstractDatasetManager):

    def __init__(self):
        super().__init__()
        self.camera_intrinsics_dict = {}
        self.camera_extrinsics_dict = {}
        self.image_names = self._generate_image_names()

        self.json_manager = JsonDatasetManager()

    def _check_dict_properties_values(self, dict1, dict2):
        dict1_copy = dict1.copy()
        dict2_copy = dict2.copy()

        dict1_copy["frames"] = []
        dict2_copy["frames"] = []

        if set(dict1_copy.keys()) == set(dict2_copy.keys()):
            for key in dict1.keys():
                if dict1_copy[key] != dict2_copy[key]:
                    raise Exception("Dictionaries do not have the same values!")
        else:
            raise Exception("Dictionaries do not have the same properties!")

    def merge_datasets(self, single_scenario) -> None:
        train_datasets_paths = single_scenario["train_datasets_paths"]
        fused_dataset_path = single_scenario["fused_dataset_path"]
        fused_dataset_folder_name = single_scenario["fused_dataset_folder_name"]

        try:
            all_paths = train_datasets_paths.copy()
            all_paths.append(fused_dataset_path)
            self._check_paths_exist(all_paths)

            train_datasets_paths_images, train_datasets_paths_transforms = self._check_dataset_structure(
                train_datasets_paths)
        except Exception as e:
            print("\033[91mIISNeRF ERROR\033[0m: ", e)
            return

        self._create_folder(fused_dataset_path + "\\" + fused_dataset_folder_name)
        self._create_folder(fused_dataset_path + "\\" + fused_dataset_folder_name + "\\train")
        self._create_folder(fused_dataset_path + "\\" + fused_dataset_folder_name + "\\train\\images")

        order: int = 0
        fused_dataset_images_complete_path = fused_dataset_path + "\\" + fused_dataset_folder_name + "\\train\\images"
        fused_dataset_transforms_complete_path = fused_dataset_path + "\\" + fused_dataset_folder_name + "\\train"

        self.json_manager.open_json_file(train_datasets_paths_transforms[0])
        camera_dict = self.json_manager.get_camera_dict()
        camera_frames = []

        try:
            for index in range(len(train_datasets_paths_transforms)):
                self.json_manager.open_json_file(train_datasets_paths_transforms[index])
                camera_dict2 = self.json_manager.get_camera_dict()
                self._check_dict_properties_values(camera_dict, camera_dict2)
        except Exception as e:
            print("\033[91mIISNeRF ERROR\033[0m: ", e)
            return

        for index, train_dataset in enumerate(train_datasets_paths_images):
            print(train_datasets_paths_images)
            file_number: int = 0
            if os.path.exists(train_dataset):
                print("train dataset exists")
                self.json_manager.open_json_file(train_datasets_paths_transforms[index])

                for i, jpgfile in enumerate(glob.iglob(os.path.join(train_dataset, "*.jpg"))):
                    file_path = fused_dataset_images_complete_path + "\\" + self.image_names[order] + ".jpg"
                    shutil.copy(jpgfile, file_path)
                    self.json_manager.change_frame_file_path("images\\" + self.image_names[order] + ".jpg", file_number)
                    order = order + 1
                    file_number = file_number + 1

                camera_frames = camera_frames + self.json_manager.get_frames()
            else:
                print("not exists")

        camera_dict["frames"] = camera_frames
        self.json_manager.write_json_file(camera_dict, fused_dataset_transforms_complete_path + r"\transforms.json")

    def create_test_dataset(self, single_scenario) -> None:
        full_fused_dataset_path = single_scenario["full_fused_dataset_path"]
        fused_dataset_folder_name = single_scenario["fused_dataset_folder_name"]
        test_images_count = single_scenario["test_images_count"]
        aabb_scale_values = single_scenario["aabb_scale_values"]
        self._check_paths_exist([full_fused_dataset_path])

        self._create_folder(full_fused_dataset_path + "\\test")

        src_path = full_fused_dataset_path + "\\train\\transforms.json"
        dst_path = full_fused_dataset_path + "\\test\\transforms.json"
        shutil.copy(src_path, dst_path)

        self.json_manager.open_json_file(dst_path)
        frames = self.json_manager.get_frames()
        if test_images_count <= len(frames):
            frames = frames[:test_images_count]
            camera_dict = self.json_manager.get_camera_dict()
            camera_dict["frames"] = frames
            self.json_manager.write_json_file(camera_dict, dst_path)

        for aabb_scale in aabb_scale_values:
            self._create_folder(full_fused_dataset_path + rf"\test\images_aabb_scale_{str(aabb_scale)}")
            subprocess.run(
                ["python", r"C:\Users\mdopiriak\PycharmProjects\iis_nerf\instant-ngp\scripts\run.py", "--load_snapshot",
                 fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\nerfs\scenes\{fused_dataset_folder_name}_aabb_scale_{str(aabb_scale)}.ingp",
                 "--screenshot_transforms",
                 fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\{fused_dataset_folder_name}\test\transforms.json",
                 "--screenshot_dir",
                 fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\{fused_dataset_folder_name}\test\images_aabb_scale_{str(aabb_scale)}"])

    def create_eval_dataset(self, single_scenario):
        full_fused_dataset_path = single_scenario["full_fused_dataset_path"]
        model_path = single_scenario["model_path"]
        trajectory_path = single_scenario["trajectory_path"]
        eval_dataset_folder_name = single_scenario["eval_dataset_folder_name"]

        self._create_folder(full_fused_dataset_path + "\\eval")
        eval_folder_path = full_fused_dataset_path + "\\eval\\" + eval_dataset_folder_name
        eval_folder_images_path = eval_folder_path + "\\images"
        self._create_folder(eval_folder_path)
        self._create_folder(eval_folder_images_path)
        eval_folder_transforms_path = eval_folder_path + "\\transforms.json"
        shutil.copy(trajectory_path, eval_folder_transforms_path)

        subprocess.run(
            ["python", r"C:\Users\mdopiriak\PycharmProjects\iis_nerf\instant-ngp\scripts\run.py",
             "--load_snapshot", model_path,
             "--screenshot_transforms", trajectory_path,
             "--screenshot_dir", eval_folder_images_path])

    def create_ipframe_dataset(self, single_scenario):

        iframe_dataset_path = single_scenario["iframe_dataset_path"]
        pframe_dataset_path = single_scenario["pframe_dataset_path"]
        full_fused_dataset_path = single_scenario["full_fused_dataset_path"]
        ipframe_dataset_folder_name = single_scenario["ipframe_dataset_folder_name"]

        order: int = 0

        video_path = full_fused_dataset_path + "\\videos"
        self._create_folder(video_path)

        self._create_folder(video_path + "\\" + ipframe_dataset_folder_name)
        image_path = video_path + "\\" + ipframe_dataset_folder_name + "\\images"  # add resolution to folder name
        self._create_folder(image_path)

        iframes = [jpgfile for jpgfile in glob.iglob(os.path.join(iframe_dataset_path, "*.jpg"))]
        pframes = [jpgfile for jpgfile in glob.iglob(os.path.join(pframe_dataset_path, "*.jpg"))]

        iframes_length = len(iframes)

        for i in range(iframes_length):
            full_video_path = image_path + "\\" + ipframe_dataset_folder_name + self.image_names[order] + ".jpg"
            shutil.copy(iframes[i], full_video_path)

            order = order + 1

            full_video_path = image_path + "\\" + ipframe_dataset_folder_name + self.image_names[order] + ".jpg"
            shutil.copy(pframes[i], full_video_path)

            order = order + 1
