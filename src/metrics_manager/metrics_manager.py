import math
import numpy as np


class MetricsManager:
    def __init__(self):
        pass

    @staticmethod
    def calculate_psnr(first_image, second_image):
        # img1 and img2 have range [0, 255]
        first_image = first_image.astype(np.float64)
        second_image = second_image.astype(np.float64)
        mse = np.mean((first_image - second_image) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))
