import os
import re
import glob
import subprocess

from file_manager.file_manager import FileManager


class VideoManager:
    def __init__(self):
        self.image_names = self._generate_image_names()
        self.file_manager = FileManager()

    def _generate_image_names(self):
        image_names = []
        for i in range(1, 1000):
            # use the zfill method to pad zeros to the left
            image_name = str(i).zfill(3)
            image_names.append(image_name)

        return image_names

    def encode_ipframe_dataset(self, single_scenario):
        ipframe_dataset_path = single_scenario["ipframe_dataset_path"]
        encoded_videos_path = single_scenario["encoded_videos_path"]
        encoded_videos_folder_name = single_scenario["encoded_videos_folder_name"]
        preset = single_scenario["h264_cfg"]["preset"]
        crf = single_scenario["h264_cfg"]["crf"]

        encoded_videos_full_path = encoded_videos_path + "\\" + encoded_videos_folder_name
        self.file_manager.create_folder(encoded_videos_full_path)

        jpgfiles = [jpgfile for jpgfile in glob.iglob(os.path.join(ipframe_dataset_path, "*.jpg"))]
        jpgfiles_len = len(jpgfiles)
        single_jpg_file = jpgfiles[0]
        file_name = single_jpg_file.split("\\")[-1].split(".")[0] + ".jpg"
        index = single_jpg_file.find("\\" + file_name)
        new_path = single_jpg_file[:index]

        file_path = new_path + "\\" + re.sub(r'\d+', '%3d', file_name)
        order: int = 1

        for i in range(jpgfiles_len):
            encoded_video_path = encoded_videos_full_path + fr"\encoded_video{i + 1}.h264"

            subprocess.run(
                fr"ffmpeg -framerate 1 -start_number {order} -i {file_path} -frames:v 2 -c:v libx264 -preset {preset} -crf {crf} -r 1 {encoded_video_path}")

            order = order + 2
