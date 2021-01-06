from typing import Tuple

from scipy import signal as sg
from skimage.feature import peak_local_max
import numpy as np


def find_tfl_lights(c_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect candidates for TFL lights.
    """
    kernel_r = np.array([[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
                         [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
                         [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
                         [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
                         [-0.5, -0.5, -0.5], [1, 1, -0.5],
                         [1, 2, 1], [1, 2, 1],
                         [1, 1, 1], [1, 1, 1]])
    kernel_g = np.array([[1, 1, 1], [1, 1, 1],
                         [1, 2, 1],  [1, 2, 1],
                         [1, 1, -0.5], [1, 1, -0.5],
                         [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
                         [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
                         [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
                         [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
                         [-0.5, -0.5, -0.5]])

    im_red = c_image[:, :, 0]
    im_green = c_image[:, :, 1]

    grad_red = sg.convolve2d(im_red, kernel_r, mode='same')
    grad_green = sg.convolve2d(im_green, kernel_g, mode='same')
    
    coordinates_red = peak_local_max(grad_red, min_distance=20, num_peaks=10)
    coordinates_green = peak_local_max(grad_green, min_distance=20, num_peaks=10)
    
    x_red, y_red = coordinates_red[:, -1], coordinates_red[:, 0]
    x_green, y_green = coordinates_green[:, -1], coordinates_green[:, 0]
    
    return x_red, y_red, x_green, y_green
