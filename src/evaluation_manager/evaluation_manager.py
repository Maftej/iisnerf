import os
import re
import cv2
import glob
import subprocess
import skimage.metrics
from skimage import metrics

from dataset_manager.json_manager import JsonDatasetManager


class EvaluationManager:

    def __init__(self):
        self.json_manager = JsonDatasetManager()

    def evaluate_models(self, single_scenario):
        full_fused_dataset_path = single_scenario["full_fused_dataset_path"]
        eval_models_full_file_path = single_scenario["eval_models_full_file_path"]

        train_dataset_path = full_fused_dataset_path + r"\train\images"
        reference_images = [cv2.imread(filename) for filename in glob.iglob(os.path.join(train_dataset_path, "*.jpg"))]

        aabb_scale = 1
        psnr_values = []
        ssim_values = []

        while True:
            test_dataset_path = full_fused_dataset_path + fr"\test\images_aabb_scale_{aabb_scale}"
            nerf_images = [cv2.imread(filename) for filename in glob.iglob(os.path.join(test_dataset_path, "*.jpg"))]
            len_nerf_images = len(nerf_images)
            iter_psnr_values = []
            iter_ssim_values = []

            for i, reference_image in enumerate(reference_images):
                if i >= len_nerf_images:
                    break
                nerf_image = nerf_images[i]
                psnr = metrics.peak_signal_noise_ratio(reference_image, nerf_image)
                ssim = metrics.structural_similarity(reference_image, nerf_image, multichannel=True, channel_axis=2)
                iter_psnr_values.append(round(psnr, 2))
                iter_ssim_values.append(round(ssim, 2))

                # lpips_values.append(lpips_metric.forward(torch.from_numpy(reference_image), torch.from_numpy(nerf_image)).item())
            psnr_values.append(iter_psnr_values)
            ssim_values.append(iter_ssim_values)

            print()
            print("**********")
            print("AABB_SCALE=", aabb_scale)
            print("**********")
            print()

            aabb_scale = aabb_scale * 2
            if aabb_scale > 128:
                break

        data = {
            "psnr": psnr_values,
            "ssim": ssim_values
        }
        self.json_manager.write_json_file(eval_models_full_file_path, data)

    def eval_trajectory(self, single_scenario):
        output_file_path = str(single_scenario["output_file_path"])
        synthetic_dataset_path = single_scenario["synthetic_dataset_path"]
        nerf_dataset_path = single_scenario["nerf_dataset_path"]
        file_format = single_scenario["file_format"] if "file_format" in single_scenario else "jpg"

        nerf_images = [cv2.imread(filename) for filename in
                       glob.iglob(os.path.join(nerf_dataset_path, "*." + file_format))]
        reference_images = [cv2.imread(filename) for filename in
                            glob.iglob(os.path.join(synthetic_dataset_path, "*." + file_format))]
        len_reference_images = len(reference_images)
        len_nerf_images = len(nerf_images)

        if len_reference_images != len_nerf_images:
            print("Datasets sizes are not equal!")
            return

        psnr_values = []
        ssim_values = []
        mse_values = []

        for i, reference_image in enumerate(reference_images):
            psnr = metrics.peak_signal_noise_ratio(reference_image, nerf_images[i])
            ssim = metrics.structural_similarity(reference_image, nerf_images[i], multichannel=True, channel_axis=2)
            mse = skimage.metrics.mean_squared_error(reference_image, nerf_images[i])
            mse_values.append(mse)
            psnr_values.append(round(psnr, 2))
            ssim_values.append(round(ssim, 2))

            print("\n*****")
            print("ITERATION=", i)
            print("MSE=", mse)
            print("PSNR=", round(psnr, 2))
            print("SSIM=", round(ssim, 2))
            print("*****\n")

        data = {
            "psnr": psnr_values,
            "ssim": ssim_values
        }

        print("AVG MSE=", round(sum(mse_values) / len(mse_values), 2))
        print("AVG PSNR=", round(sum(psnr_values) / len(psnr_values), 2))
        print("AVG SSIM=", round(sum(ssim_values) / len(ssim_values), 2))
        print(data)

        self.json_manager.write_json_file(output_file_path, data)

    def evaluate_encoded_ipframe(self, single_scenario):
        ipframe_encoded_videos_path = single_scenario["ipframe_encoded_videos_path"]
        pkt_size_file_name = single_scenario["pkt_size_file_name"]

        h264files = [h264file for h264file in glob.iglob(os.path.join(ipframe_encoded_videos_path, "*.h264"))]
        h264files_len = len(h264files)

        order: int = 1
        encoded_videos_data = []

        for i in range(h264files_len):
            file_path = re.sub(r"\d+(?=\.\w+$)", str(order), h264files[0])
            print(file_path)
            filename = file_path.split("\\")[-1]

            subprocess_output = subprocess.check_output(
                fr"ffprobe -show_frames {file_path}")

            order = order + 1

            subprocess_output_str = subprocess_output.decode('utf-8')
            subprocess_output_str_lines = subprocess_output_str.split('\n')
            packets = []
            for s in subprocess_output_str_lines:
                if s.startswith("pkt_size"):
                    # Found a string that starts with "pkt_size"
                    number_str = s.replace("pkt_size=", "")
                    number = int(number_str)
                    packets.append(number)

            encoded_videos_data.append(packets)

        full_pkt_size_filename = ipframe_encoded_videos_path + "\\" + pkt_size_file_name
        self.json_manager.write_json_file(full_pkt_size_filename, encoded_videos_data)

    def evaluate_encoded_h264(self, single_scenario):
        h264_encoded_video_path = single_scenario["h264_encoded_video_path"]
        h264_file_name = single_scenario["h264_file_name"]
        output_file_name = single_scenario["output_file_name"]

        h264_encoded_video_full_path = h264_encoded_video_path + "\\" + h264_file_name

        encoded_videos_data = []
        subprocess_output = subprocess.check_output(
            fr"ffprobe -show_frames {h264_encoded_video_full_path}")

        subprocess_output_str = subprocess_output.decode('utf-8')
        subprocess_output_str_lines = subprocess_output_str.split('\n')
        packets = []
        for s in subprocess_output_str_lines:
            if s.startswith("pkt_size"):
                # Found a string that starts with "pkt_size"
                number_str = s.replace("pkt_size=", "")
                number = int(number_str)
                packets.append(number)
            if s.startswith("pict_type"):
                pict_type_str = s.replace("pict_type=", "")
                packets.append(pict_type_str[0])

        encoded_videos_data = {
            'data': packets
        }
        print(encoded_videos_data)
        output_filename_full_path = h264_encoded_video_path + "\\" + output_file_name
        self.json_manager.write_json_file(output_filename_full_path, encoded_videos_data)
