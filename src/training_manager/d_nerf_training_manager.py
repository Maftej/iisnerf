import subprocess

from training_manager.abstract_training_manager import AbstractTrainingManager


class DNeRFTrainingManager(AbstractTrainingManager):
    def train(self, single_scenario):
        config_path = single_scenario["config_path"]

        subprocess.run(
            ["python", r"C:\Users\mdopiriak\PycharmProjects\iis_nerf\src\d_nerf\D_NeRF\run_dnerf.py", "--config",
             config_path])
