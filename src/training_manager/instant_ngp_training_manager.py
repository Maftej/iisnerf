import subprocess

from dataset_manager.json_manager import JsonDatasetManager
from training_manager.abstract_training_manager import AbstractTrainingManager


class InstantNgpTrainingManager(AbstractTrainingManager):
    def __init__(self):
        self.json_manager = JsonDatasetManager()

    def train(self, single_scenario):
        # n_steps, aabb_scale_values, full_fused_dataset_path, fused_dataset_folder_name
        n_steps = single_scenario["n_steps"]
        aabb_scale_values = single_scenario["aabb_scale_values"]
        full_fused_dataset_path = single_scenario["full_fused_dataset_path"]
        fused_dataset_folder_name = single_scenario["fused_dataset_folder_name"]

        full_dataset_path = full_fused_dataset_path + r"\train"
        camera_dict_path = full_dataset_path + r"\transforms.json"
        for aabb_scale in aabb_scale_values:
            self.json_manager.open_json_file(camera_dict_path)
            self.json_manager.change_aabb_scale(aabb_scale)
            self.json_manager.write_json_file(self.json_manager.get_camera_dict(), camera_dict_path)

            subprocess.run(
                ["python", r"C:\Users\mdopiriak\PycharmProjects\iis_nerf\instant-ngp\scripts\run.py", "--scene",
                 full_dataset_path, "--train", "--n_steps", str(n_steps), "--save_snapshot",
                 fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\nerfs\scenes\{fused_dataset_folder_name}_aabb_scale_{str(aabb_scale)}.ingp"])
