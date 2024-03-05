from file_manager.file_manager import FileManager


class JsonDatasetManager:
    def __init__(self):
        self.file_manager = FileManager()
        self.camera_dict = None

    def open_json_file(self, file):
        self.camera_dict = self.file_manager.open_file(file)

    def get_camera_dict(self):
        return self.camera_dict

    def get_camera_angle_x(self):
        return self.camera_dict["camera_angle_x"]

    def get_frames(self):
        return self.camera_dict["frames"]

    def write_json_file(self, file_path, data):
        self.file_manager.write_file(file_path, data)

    def change_aabb_scale(self, aabb_scale: int):
        self.camera_dict["aabb_scale"] = aabb_scale

    def change_frames_folder(self, folder: str) -> None:
        frames = self.camera_dict["frames"]
        if self.camera_dict is not None:
            for frame in frames:
                file_path = frame["file_path"].split("\\")
                if len(file_path) > 1:
                    frame["file_path"] = folder + "\\" + file_path[1]
                else:
                    print("Error: frames file path!")
        else:
            print("Error: File is not opened!")

    def change_frame_file_path(self, file_path: str, index: int) -> None:
        frames = self.camera_dict["frames"]
        if self.camera_dict is not None:
            temp = frames[index]
            frames[index]["file_path"] = file_path
        else:
            print("Error: File is not opened!")

