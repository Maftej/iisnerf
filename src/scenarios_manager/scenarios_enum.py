from enum import Enum


class ScenariosEnum(Enum):
    MERGE_DATASETS = "MERGE_DATASETS"
    TRAIN = "TRAIN"
    TEST_DATASET = "TEST_DATASET"
    EVAL_MODELS = "EVAL_MODELS"
    EVAL_DATASET = "EVAL_DATASET"
    IPFRAME_DATASET = "IPFRAME_DATASET"
    ENCODE_IPFRAME_DATASET = "ENCODE_IPFRAME_DATASET"
    EVAL_ENCODED_IPFRAME = "EVAL_ENCODED_IPFRAME"
    EVAL_TRAJECTORY = "EVAL_TRAJECTORY"
    EVAL_ENCODED_H264 = "EVAL_ENCODED_H264"

    PLOT_ALL_MAPS = "PLOT_ALL_MAPS"
    PLOT_DATA = "PLOT_DATA"
