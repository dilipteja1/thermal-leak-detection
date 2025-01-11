import os
from typing import Optional
from nptyping import Array
import pathlib
import numpy as np
import cv2
from flyr.camera_metadata import CameraMetadata
from constants import Camera
import exiftool
import datetime
from collections import defaultdict
from flyr import unpack

class DataPoint():
    def __init__(self, thermal_image_path):
        self.thermal_image_path = thermal_image_path
        self.visual_image_path = self.get_visual_image_path
        self.thermogram = unpack(self.thermal_image_path)
        self.cm = CameraMetadata(self.thermal_image_path)
        self.image_path_parts = pathlib.Path(self.thermal_image_path).parts

    @property
    def get_thermal(self) -> Optional[Array[np.uint8, ..., ..., 3]]:
        return self.thermogram.render(palette='grayscale')

    @property
    def get_visual(self):
        """ visual image"""
        return cv2.imread(self.visual_image_path)

    @property
    def get_camera(self) -> [str, None]:
        """ Get the camera used to capture the thermal image """
        return Camera.E50 if Camera.E50 in self.cm.model else Camera.EDGEPRO

    @property
    def complete_meta_data(self) -> [defaultdict]:
        with exiftool.ExifToolHelper() as et:
            meta_data = et.get_metadata(self.thermal_image_path)
        return defaultdict(str, meta_data[0])

    @property
    def image_type(self) -> [str, None]:
        """
            get the type of the image
        """
        image_type = self.complete_meta_data["APP1:RawThermalImageType"]
        return None if type is None else image_type

    @property
    def get_image_name(self):
        return self.image_path_parts[len(self.image_path_parts)-1]
    @property
    def get_window(self)-> [str, None]:
        """gets the window name from the folder structure"""
        return self.image_path_parts[len(self.image_path_parts)-2]

    @property
    def get_opening(self) -> [str, None]:
        """gets the opening of the window for annotation purposes"""
        return self.image_path_parts[len(self.image_path_parts)-3]
    
    @property
    def get_date(self) -> [datetime,None]:
        """gets the date of the image taken from the folder structure"""
        return self.image_path_parts[len(self.image_path_parts) - 4]

    @property
    def get_visual_image_path(self):
        """
            Picks an IR image file and get the corresponding DC image file
            and visual image. this is for E50 camera
            #TODO make a diverse data point (text + images)
        Returns:
            data point tuple
        """
        folder_name = os.path.dirname(self.thermal_image_path)
        file_name, extension = os.path.splitext(os.path.basename(self.thermal_image_path))
        thermal_image_number = int(file_name.split('_')[1])
        visual_image_number = thermal_image_number + 1
        visual_image_name = "DC_" + str(visual_image_number) + ".jpg"
        visual_image_file = os.path.join(folder_name, visual_image_name)
        if os.path.exists(visual_image_file):
            return str(visual_image_file)
        else:
            print(
                f"visual image for the corresponding Thermal image {os.path.basename(self.thermal_image_path)} not found")
            return None
