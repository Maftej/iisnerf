import os
from abc import ABC, abstractmethod

from scenarios_manager.dataset_type_enum import DatasetTypeEnum


class AbstractDatasetManager(ABC):
    @abstractmethod
    def merge_datasets(self, single_scenario, dataset_type=DatasetTypeEnum.TRAIN.value) -> None:
        pass

    def _generate_image_names(self):
        image_names = []
        for i in range(1, 1000):
            # use the zfill method to pad zeros to the left
            image_name = str(i).zfill(3)
            image_names.append(image_name)

        return image_names

    def _create_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            print("folder path exists")

    def _check_paths_exist(self, paths: list[str]):
        for path in paths:
            if not os.path.exists(path):
                raise Exception("Path does not exists! Path=", path)

    def _check_dataset_structure(self, train_datasets_paths: list[str]) -> tuple[list[str], list[str]]:
        train_datasets_paths_images = []
        train_datasets_paths_transforms = []

        for train_datasets_path in train_datasets_paths:
            dir_list = os.listdir(train_datasets_path)
            if len(dir_list) != 2:
                raise Exception("Dataset has to contain folder and json file")

            for dir in dir_list:
                if os.path.isdir(train_datasets_path + "\\" + dir) == True:
                    train_datasets_paths_images.append(train_datasets_path + "\\" + dir)
                else:
                    train_datasets_paths_transforms.append(train_datasets_path + "\\" + dir)
                    extension = os.path.splitext(dir)[1]
                    if extension != ".json":
                        raise Exception("Dataset does not contain json file!")

        return train_datasets_paths_images, train_datasets_paths_transforms
