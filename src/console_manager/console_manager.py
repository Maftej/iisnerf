import argparse

from scenarios_manager.scenarios_enum import ScenariosEnum
from scenarios_manager.dataset_type_enum import DatasetTypeEnum
from scenarios_manager.nerf_variants_enum import NeRFVariantsEnum


class ConsoleManager:
    def __init__(self):
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Run IIS-NeRF")

        parser.add_argument("--nerf_variant",
                            choices=[NeRFVariantsEnum.D_NERF.value, NeRFVariantsEnum.INSTANT_NGP.value],
                            help="")

        parser.add_argument("--scenario", choices=[ScenariosEnum.MERGE_DATASETS.value, ScenariosEnum.TRAIN.value,
                                                   ScenariosEnum.TEST_DATASET.value, ScenariosEnum.EVAL_DATASET.value,
                                                   ScenariosEnum.IPFRAME_DATASET.value,
                                                   ScenariosEnum.ENCODE_IPFRAME_DATASET.value,
                                                   ScenariosEnum.EVAL_ENCODED_IPFRAME.value,
                                                   ScenariosEnum.EVAL_MODELS.value,
                                                   ScenariosEnum.EVAL_TRAJECTORY.value,
                                                   ScenariosEnum.EVAL_ENCODED_H264.value,
                                                   ScenariosEnum.PLOT_ALL_MAPS.value,
                                                   ScenariosEnum.PLOT_DATA.value],
                            help="")

        parser.add_argument("--dataset_type", default=None,
                            choices=[DatasetTypeEnum.TRAIN.value, DatasetTypeEnum.TEST.value,
                                     DatasetTypeEnum.VAL.value],
                            help="")

        parser.add_argument("--scenario_path", default="",
                            help="")

        return parser.parse_args()
