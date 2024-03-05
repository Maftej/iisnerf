import os
import cv2
import glob
from skimage import metrics

from file_manager.file_manager import FileManager


class DNeRFEvaluationManager:
    def __init__(self):
        self.file_manager = FileManager()
    def evaluate_train_dataset(self, single_scenario):
        output_file_path = single_scenario["output_file_path"]
        synthetic_dataset_path = single_scenario["synthetic_dataset_path"]
        nerf_dataset_path = single_scenario["nerf_dataset_path"]

        nerf_images = [cv2.imread(filename) for filename in glob.iglob(os.path.join(nerf_dataset_path, "*.jpg"))]
        reference_images = [cv2.imread(filename) for filename in
                            glob.iglob(os.path.join(synthetic_dataset_path, "*.jpg"))]
        len_reference_images = len(reference_images)
        len_nerf_images = len(nerf_images)

        if len_reference_images != len_nerf_images:
            print("Datasets sizes are not equal!")
            return

        psnr_values = []
        ssim_values = []

        for i, reference_image in enumerate(reference_images):
            psnr = metrics.peak_signal_noise_ratio(reference_image, nerf_images[i])
            ssim = metrics.structural_similarity(reference_image, nerf_images[i], multichannel=True, channel_axis=2)
            psnr_values.append(round(psnr, 2))
            ssim_values.append(round(ssim, 2))

            print("ITERATION=", i)

        data = {
            "psnr": psnr_values,
            "ssim": ssim_values
        }
        self.file_manager.write_json_file(output_file_path, data)
