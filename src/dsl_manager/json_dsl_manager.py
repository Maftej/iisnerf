import json

from dsl_manager.dsl_manager import DslManager
from video_manager.video_manager import VideoManager
from scenarios_manager.scenarios_enum import ScenariosEnum
from plot_manager.d_nerf_plot_manager import DNeRFPlotManager
from scenarios_manager.nerf_variants_enum import NeRFVariantsEnum
from evaluation_manager.evaluation_manager import EvaluationManager
from dataset_manager.d_nerf_dataset_manager import DNeRFDatasetManager
from training_manager.d_nerf_training_manager import DNeRFTrainingManager
from dataset_manager.instant_ngp_dataset_manager import InstantNGPDatasetManager
from training_manager.instant_ngp_training_manager import InstantNgpTrainingManager


class JsonDslManager(DslManager):
    def __init__(self):
        self.video_manager = VideoManager()
        self.evaluation_manager = EvaluationManager()
        self.instant_ngp_dataset_manager = InstantNGPDatasetManager()
        self.d_nerf_dataset_manager = DNeRFDatasetManager()
        self.instant_ngp_training_manager = InstantNgpTrainingManager()
        self.d_nerf_training_manager = DNeRFTrainingManager()

    def run_all_scenarios(self):
        data = None
        with open(self.scenario_file_path, 'r') as f:
            # Load the JSON data into a Python dictionary
            data = json.load(f)

        scenarios = data["scenarios"]
        print(scenarios)
        scenario_fused_datasets_data = None
        scenario_train_nerf_data = None
        scenario_test_nerf_data = None
        scenario_eval_nerf_dataset_data = None
        scenario_ipframe_dataset_data = None
        scenario_encode_ipframe_dataset_data = None

        for scenario in scenarios:
            if scenario == ScenariosEnum.MERGE_DATASETS:
                scenario_fused_datasets_data = scenario
            elif scenario == ScenariosEnum.TRAIN:
                scenario_train_nerf_data = scenario
            elif scenario == ScenariosEnum.TEST_DATASET:
                scenario_test_nerf_data = scenario
            elif scenario == ScenariosEnum.EVAL_DATASET:
                scenario_eval_nerf_dataset_data = scenario
            elif scenario == ScenariosEnum.IPFRAME_DATASET:
                scenario_ipframe_dataset_data = scenario
            elif scenario == ScenariosEnum.ENCODE_IPFRAME_DATASET:
                scenario_encode_ipframe_dataset_data = scenario

        if scenario_fused_datasets_data is not None:
            print(scenario_fused_datasets_data)

        if scenario_train_nerf_data is not None:
            print(scenario_train_nerf_data)

        if scenario_test_nerf_data is not None:
            print(scenario_test_nerf_data)

        if scenario_eval_nerf_dataset_data is not None:
            print(scenario_eval_nerf_dataset_data)

        if scenario_ipframe_dataset_data is not None:
            print(scenario_ipframe_dataset_data)

        if scenario_encode_ipframe_dataset_data is not None:
            print(scenario_encode_ipframe_dataset_data)

    def _switch_scenario_instant_ngp(self, scenario, single_scenario):
        switcher = {
            ScenariosEnum.MERGE_DATASETS.value: self.instant_ngp_dataset_manager.merge_datasets,
            ScenariosEnum.TRAIN.value: self.instant_ngp_training_manager.train,
            ScenariosEnum.TEST_DATASET.value: self.instant_ngp_dataset_manager.create_test_dataset,
            ScenariosEnum.EVAL_DATASET.value: self.instant_ngp_dataset_manager.create_eval_dataset,
            ScenariosEnum.IPFRAME_DATASET.value: self.instant_ngp_dataset_manager.create_ipframe_dataset,
            ScenariosEnum.ENCODE_IPFRAME_DATASET.value: self.video_manager.encode_ipframe_dataset,
            ScenariosEnum.EVAL_ENCODED_IPFRAME.value: self.evaluation_manager.evaluate_encoded_ipframe,
            ScenariosEnum.EVAL_ENCODED_H264.value: self.evaluation_manager.evaluate_encoded_h264,
            ScenariosEnum.EVAL_MODELS.value: self.evaluation_manager.evaluate_models,
            ScenariosEnum.EVAL_TRAJECTORY.value: self.evaluation_manager.eval_trajectory,
        }

        func = switcher.get(scenario, lambda: "Invalid scenario")
        return func(single_scenario)

    def _switch_scenario_d_nerf(self, scenario, single_scenario, dataset_type=None):
        switcher = {
            ScenariosEnum.MERGE_DATASETS.value: self.d_nerf_dataset_manager.merge_datasets,
            ScenariosEnum.TRAIN.value: self.d_nerf_training_manager.train,
            ScenariosEnum.EVAL_DATASET.value: self.d_nerf_dataset_manager.generate_dataset,
            ScenariosEnum.EVAL_TRAJECTORY.value: self.evaluation_manager.eval_trajectory,
            ScenariosEnum.PLOT_ALL_MAPS.value: DNeRFPlotManager.plot_all_maps,
            ScenariosEnum.PLOT_DATA.value: DNeRFPlotManager.plot_psnr_ssim_data
        }

        func = switcher.get(scenario, lambda: "Invalid scenario")

        if dataset_type is not None:
            dataset_type_lower = dataset_type.lower()
            return func(single_scenario, dataset_type_lower)
        else:
            return func(single_scenario)

    def run_single_scenario(self, nerf_variant, scenario, scenario_path, dataset_type=None):
        data = None
        with open(scenario_path, 'r') as f:
            # Load the JSON data into a Python dictionary
            data = json.load(f)

        scenarios = data["scenarios"]
        single_scenario = next(filter(lambda scenario_enum: scenario == scenario_enum["scenario"], scenarios), None)

        if nerf_variant == NeRFVariantsEnum.INSTANT_NGP.value:
            self._switch_scenario_instant_ngp(scenario, single_scenario)
        elif nerf_variant == NeRFVariantsEnum.D_NERF.value:
            if scenario == ScenariosEnum.MERGE_DATASETS.value and dataset_type is not None:
                self._switch_scenario_d_nerf(scenario, single_scenario, dataset_type)
            else:
                self._switch_scenario_d_nerf(scenario, single_scenario)
