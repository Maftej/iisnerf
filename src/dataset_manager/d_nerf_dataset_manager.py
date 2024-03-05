import os
import cv2
import glob
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt

from render_manager.render_manager import RenderManager
from dataset_manager.json_manager import JsonDatasetManager
from scenarios_manager.dataset_type_enum import DatasetTypeEnum
from dataset_manager.abstract_dataset_manager import AbstractDatasetManager
from nerf_variants.d_nerf.run_dnerf import render_path_all, config_parser, create_nerf


class DNeRFDatasetManager(AbstractDatasetManager):
    def __init__(self):
        super().__init__()
        self.json_dataset_manager = JsonDatasetManager()
        self.image_names = self._generate_image_names()
        self.render_manager = RenderManager()

    def generate_dataset(self, single_scenario):
        full_fused_dataset_path = single_scenario["full_fused_dataset_path"]
        model_path = single_scenario["model_path"]
        trajectory_path = single_scenario["trajectory_path"]
        eval_dataset_folder_name = single_scenario["eval_dataset_folder_name"]

        json_dataset_manager = JsonDatasetManager()
        json_dataset_manager.open_json_file(trajectory_path)
        camera_dict = json_dataset_manager.get_camera_dict()
        # print("@@#=", len(camera_dict["frames"]))

        camera_frames = camera_dict["frames"]

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DEVICE=", device)
        # get config file
        config_file = fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\src\d_nerf\D_NeRF\configs\arm_robot.txt"
        parser = config_parser()
        args = parser.parse_args(f'--config {config_file}')

        # set render params
        hwf = [800, 800, 1113]
        _, render_kwargs_test, _, _, _ = create_nerf(args)
        render_kwargs_test.update({'near': 2., 'far': 6.})

        full_path = full_fused_dataset_path + "\\" + eval_dataset_folder_name
        self._create_folder(full_path)

        for i, camera_frame in enumerate(camera_frames):
            # print("TRANSFORMATION_MATRIX=", camera_frame["transform_matrix"])
            # print()
            # print("TIME=", camera_frame["time"])
            # print("\n")
            render_poses = torch.tensor([camera_frame["transform_matrix"]])
            # render_poses = torch.unsqueeze(pose_spherical(180, 20, 4.0), 0).to(device)
            print("RENDER_POSES=", render_poses)
            # time = camera_frame["time"]
            time = camera_frame['time'] if 'time' in camera_frame else float(i) / (len(camera_dict['frames'][::1]) - 1)
            print("TIME=", time)
            render_times = torch.Tensor([time]).to(device)

            with torch.no_grad():
                rgbs, disps, accs, depths = render_path_all(render_poses, render_times, hwf, args.chunk,
                                                              render_kwargs_test,
                                                              render_factor=args.render_factor)

            # rgbs = to8b(rgbs)
            disp = disps[0]
            acc = accs[0]

            kernel = np.ones((7, 7), np.uint8)
            mg_erosion = cv2.erode(acc, kernel, iterations=2)
            img_dilation = cv2.dilate(mg_erosion, kernel, iterations=2)

            img_dilation = cv2.convertScaleAbs(img_dilation)
            th, im_gray_th_otsu = cv2.threshold(img_dilation, 128, 192, cv2.THRESH_OTSU)

            masked = cv2.bitwise_and(disp, disp, mask=im_gray_th_otsu)

            # Create a colormap for the heatmap
            cmap = plt.get_cmap('viridis')

            # Normalize the disparity map
            normalized_disparity = cv2.normalize(masked, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                                 dtype=cv2.CV_32F)

            # Apply the colormap to the normalized disparity map
            heatmap = (cmap(normalized_disparity) * 255).astype(np.uint8)
            cv2.imwrite(full_path + "\\" + self.image_names[i] + ".png",
                        cv2.cvtColor(heatmap.astype(np.uint8), cv2.COLOR_BGR2RGB))

            # cv2.imwrite(full_path + "\\" + self.image_names[i] + ".png", cv2.cvtColor(rgbs[0], cv2.COLOR_BGR2RGB))

    def merge_datasets(self, single_scenario, dataset_type=DatasetTypeEnum.TRAIN.value) -> None:
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
        self._create_folder(fused_dataset_path + "\\" + fused_dataset_folder_name + "\\" + dataset_type)
        self._create_folder(fused_dataset_path + "\\" + fused_dataset_folder_name + "\\" + dataset_type + "\\images")

        order: int = 0
        fused_dataset_images_complete_path = fused_dataset_path + "\\" + fused_dataset_folder_name + "\\" + dataset_type + "\\images"
        fused_dataset_transforms_complete_path = fused_dataset_path + "\\" + fused_dataset_folder_name + "\\" + dataset_type

        self.json_dataset_manager.open_json_file(train_datasets_paths_transforms[0])
        camera_dict = self.json_dataset_manager.get_camera_dict()
        camera_frames = []

        json_dataset_managers = []
        train_datasets = []

        for index, train_dataset in enumerate(train_datasets_paths_images):
            json_dataset_managers.append(JsonDatasetManager())
            json_dataset_managers[index].open_json_file(train_datasets_paths_transforms[index])
            jpgfiles = []
            for i, jpgfile in enumerate(glob.iglob(os.path.join(train_dataset, "*.png"))):
                jpgfiles.append(jpgfile)

            train_datasets.append(jpgfiles)

        images_count = len(train_datasets[0])
        file_number: int = 0
        for i in range(images_count):
            for index, jpgfiles in enumerate(train_datasets):
                file_path = fused_dataset_images_complete_path + "\\" + self.image_names[order] + ".png"
                shutil.copy(jpgfiles[i], file_path)
                json_dataset_managers[index].change_frame_file_path(
                    dataset_type + "\\" + self.image_names[order] + ".png", file_number)
                order = order + 1
            file_number = file_number + 1

        frames_length = images_count
        for i in range(images_count):
            if i + 1 == frames_length:
                frame_time = 1
            else:
                frame_time = i / frames_length
            for json_dataset_manager in json_dataset_managers:
                single_frame = json_dataset_manager.get_frames()[i]
                camera_frames.append(self.add_time_to_frame(frame_time, single_frame))

        camera_dict["frames"] = camera_frames
        self.json_dataset_manager.write_json_file(camera_dict,
                                                  fused_dataset_transforms_complete_path + r"\transforms.json")

    def merge_npy_files(self):
        train_datasets_paths = [
            fr"C:\Users\mdopiriak\Desktop\lightfield\unused_blenderender_output\d_nerf_train_middle",
            fr"C:\Users\mdopiriak\Desktop\lightfield\unused_blenderender_output\d_nerf_train_bottom",
            fr"C:\Users\mdopiriak\Desktop\lightfield\unused_blenderender_output\d_nerf_train_top"
        ]

        fused_dataset_path = fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\arm_robot_eval_dataset"
        fused_dataset_folder_name = "d_nerf_blender_train"

        try:
            all_paths = train_datasets_paths.copy()
            all_paths.append(fused_dataset_path)
            self._check_paths_exist(all_paths)

        except Exception as e:
            print("\033[91mIISNeRF ERROR\033[0m: ", e)
            return

        self._create_folder(fused_dataset_path + "\\" + fused_dataset_folder_name)

        order: int = 0
        fused_dataset_images_complete_path = fused_dataset_path + "\\" + fused_dataset_folder_name

        train_datasets = []

        # load all images of the dataset into the array
        for index, train_dataset in enumerate(train_datasets_paths):
            jpgfiles = []
            for i, jpgfile in enumerate(glob.iglob(os.path.join(train_dataset, "*.npy"))):
                jpgfile = self.render_manager.transform_to_disparity_heatmap(jpgfile)
                jpgfiles.append(jpgfile)

            train_datasets.append(jpgfiles)

        images_count = len(train_datasets[0])
        for i in range(images_count):
            for index, jpgfiles in enumerate(train_datasets):
                file_path = fused_dataset_images_complete_path + "\\" + self.image_names[order] + ".png"
                cv2.imwrite(file_path, jpgfiles[i])
                order = order + 1

    def generate_images_from_npy_files(self):
        dataset_paths = [fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\arm_robot_eval_dataset\d_nerf_blender_val"]

        fused_dataset_path = fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\datasets\arm_robot_eval_dataset"
        fused_dataset_folder_name = "d_nerf_blender_disparity_map_val"

        try:
            all_paths = dataset_paths.copy()
            all_paths.append(fused_dataset_path)
            self._check_paths_exist(all_paths)

        except Exception as e:
            print("\033[91mIISNeRF ERROR\033[0m: ", e)
            return

        self._create_folder(fused_dataset_path + "\\" + fused_dataset_folder_name)

        order: int = 0
        fused_dataset_images_complete_path = fused_dataset_path + "\\" + fused_dataset_folder_name

        # load all images of the dataset into the array
        for index, train_dataset in enumerate(dataset_paths):
            for i, jpgfile in enumerate(glob.iglob(os.path.join(train_dataset, "*.npy"))):
                jpgfile = self.render_manager.transform_to_disparity_heatmap(jpgfile)
                file_path = fused_dataset_images_complete_path + "\\" + self.image_names[order] + ".png"
                cv2.imwrite(file_path, jpgfile)
                order = order + 1

    def add_time_to_frame(self, time: float, single_frame):
        single_frame["time"] = time
        return single_frame

    def from_blender_to_d_nerf(self, file_path, output_file_path, frame_folder):
        self.json_dataset_manager.open_json_file(file_path)
        print(self.json_dataset_manager.get_camera_angle_x())
        print(len(self.json_dataset_manager.get_frames()))
        frames_length = len(self.json_dataset_manager.get_frames())
        frames = self.json_dataset_manager.get_frames()

        for i, frame in enumerate(frames):
            # if i != 0:
            #     time = i / frames_length
            # elif i == fram
            frame["time"] = i / frames_length
            file_path = frame["file_path"].split("\\")
            if len(file_path) > 1:
                frame["file_path"] = frame_folder + "\\" + file_path[1]
            else:
                print("Error: frames file path!")

        new_json = {
            "camera_angle_x": self.json_dataset_manager.get_camera_angle_x(),
            "frames": frames
        }

        self.json_dataset_manager.write_json_file(new_json, output_file_path)
        print(new_json)


if __name__ == "__main__":
    d_nerf_dataset_manager = DNeRFDatasetManager()
    # d_nerf_dataset_manager.merge_npy_files()
    d_nerf_dataset_manager.generate_images_from_npy_files()
# file_path = r"C:\Users\mdopiriak\Desktop\blender_industrial_interrior_scenes\arm_robot_scenes\black-honey-robotic-arm\source\dynamic_robot\dynamic_robot_train\transforms_train.json"
# d_nerf_dataset_manager = DNeRFDatasetManager()
# new_json_file_path = r"C:\Users\mdopiriak\Desktop\blender_industrial_interrior_scenes\arm_robot_scenes\black-honey-robotic-arm\source\dynamic_robot\new_json_file.json"
# frame_folder = "train"
# d_nerf_dataset_manager.from_blender_to_d_nerf(file_path, new_json_file_path, frame_folder)
