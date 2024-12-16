from typing import Optional

import cv2
import numpy as np
from flyr import unpack
from nptyping import Array
from matplotlib import pyplot as plt
from dataset_processing.constants import Camera
from dataset_processing.metadata import ImageMetaData


class Thermal:
    def __init__(self, thermal_image_file, visual_image_file, camera_meta_data: [ImageMetaData, None]):
        self.thermal_image_path = thermal_image_file
        self.visual_image_path = visual_image_file
        self.thermogram = unpack(self.thermal_image_path)
        self.cm = camera_meta_data

    @property
    def get_thermal(self) -> Optional[Array[np.uint8, ..., ..., 3]]:
        return self.thermogram.render(palette='grayscale')

    @property
    def get_visual(self):
        """ visual image embedded or a seperated image"""
        if self.get_camera == Camera.EDGEPRO:
            return self.thermogram.optical_pil
        elif self.get_camera == Camera.E50:
            return cv2.imread(self.visual_image_path)
        else:
            return None

    @property
    def get_camera(self) -> [str, None]:
        """ Get the camera used to capture the thermal image """
        return Camera.E50 if Camera.E50 in self.cm.model else Camera.EDGEPRO

    # @property
    # def get_normalized_thermal(self):
    #     return self.thermogram.render(unit='fahrenheit', palette="grayscale")
    #
    # @property
    # def get_normalized_grascale_thermal(self):
    #     return cv2.cvtColor(self.get_normalized_thermal, cv2.COLOR_BGR2GRAY)
    # @property
    # def get_grayscale_thermal(self):
    #     return cv2.cvtColor(self.get_thermal, cv2.COLOR_BGR2GRAY)
