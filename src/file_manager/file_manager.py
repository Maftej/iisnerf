import os
import json


class FileManager:
    def __init__(self):
        pass

    def write_file(self, file_path, data):
        with open(str(file_path), "w") as f:
            json.dump(data, f)

    def open_file(self, path):
        with open(path, "r") as read_file:
            return json.load(read_file)

    def create_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            print("folder path exists")
