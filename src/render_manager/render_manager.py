import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from file_manager.file_manager import FileManager
from nerf_variants.d_nerf.run_dnerf_helpers import to8b
from nerf_variants.d_nerf.run_dnerf import render_path_all, config_parser, create_nerf


class RenderManager:
    def __init__(self):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DEVICE=", self.device)
        # get config file
        self.config_file = fr"C:\Users\mdopiriak\PycharmProjects\iis_nerf\src\d_nerf\D_NeRF\configs\arm_robot.txt"
        self.parser = config_parser()
        self.args = self.parser.parse_args(f'--config {self.config_file}')

        # set render params
        # self.hwf = [800, 800, 555.555]
        # self.hwf = [800, 800, 555.555]
        self.hwf = [800, 800, 1113]
        _, self.render_kwargs_test, _, _, _ = create_nerf(self.args)
        self.render_kwargs_test.update({'near': 2, 'far': 6.})
        self.file_manager = FileManager()

    def render_maps(self, time, azimuth, elevation):
        assert 0. <= time <= 1.
        assert -180 <= azimuth <= 180
        assert -180 <= elevation <= 180

        render_poses = torch.tensor([[
            [
                0.04953807219862938,
                -0.0051540168933570385,
                -0.9987589120864868,
                -3.3026156425476074
            ],
            [
                -0.9987722635269165,
                -0.00025563393137417734,
                -0.04953741282224655,
                -0.09500141441822052
            ],
            [
                0.0,
                0.9999867081642151,
                -0.005160352680832148,
                0.5105679035186768
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]
        ]])

        render_times = torch.Tensor([time]).to(self.device)

        with torch.no_grad():
            rgbs, disps, accs, depths = render_path_all(render_poses, render_times, self.hwf, self.args.chunk,
                                                          self.render_kwargs_test,
                                                          render_factor=self.args.render_factor)

        rgbs = to8b(rgbs)
        disps = to8b(disps)
        accs = to8b(accs)
        depths = to8b(depths)

        return rgbs[0], disps[0], accs[0], depths[0]

    def create_normalized_heatmap(self, map):
        normalized_map = cv2.normalize(map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_32F)

        return (normalized_map * 255).astype(np.uint8)

    def process_acc_map(self, acc):
        kernel = np.ones((7, 7), np.uint8)
        mg_erosion = cv2.erode(acc, kernel, iterations=2)
        img_dilation = cv2.dilate(mg_erosion, kernel, iterations=2)
        th, im_gray_th_otsu = cv2.threshold(img_dilation, 128, 192, cv2.THRESH_OTSU)
        masked = cv2.bitwise_and(acc, acc, mask=im_gray_th_otsu)

        return masked

    def process_disp_map(self, disp, acc):
        kernel = np.ones((7, 7), np.uint8)
        mg_erosion = cv2.erode(acc, kernel, iterations=2)
        img_dilation = cv2.dilate(mg_erosion, kernel, iterations=2)
        th, im_gray_th_otsu = cv2.threshold(img_dilation, 128, 192, cv2.THRESH_OTSU)
        masked = cv2.bitwise_and(disp, disp, mask=im_gray_th_otsu)

        return masked

    def transform_to_disparity_heatmap(self, disp):
        disparity_map = np.load(disp)
        disparity_map = np.rot90(disparity_map, 2)
        disparity_map = np.fliplr(disparity_map)

        normalized_disparity = cv2.normalize(disparity_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_32F)

        cmap = plt.get_cmap('viridis')
        heatmap = (cmap(normalized_disparity) * 255).astype(np.uint8)
        return cv2.cvtColor(heatmap.astype(np.uint8), cv2.COLOR_BGR2RGB)
