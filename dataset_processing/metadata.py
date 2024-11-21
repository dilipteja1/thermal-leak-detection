import pathlib
from flyr.camera_metadata import CameraMetadata
from dataset_processing.constants import Camera
import exiftool
import datetime
from collections import defaultdict

class ImageMetaData(CameraMetadata):
    def __init__(self, image):
        super().__init__(image)
        self.image = image
        self.image_path_parts = pathlib.Path(self.image).parts

    @property
    def complete_meta_data(self) -> [defaultdict]:
        with exiftool.ExifToolHelper() as et:
            meta_data = et.get_metadata(self.image)
        return defaultdict(str, meta_data[0])

    @property
    def image_type(self) -> [str, None]:
        """
            get the type of the image
        """
        image_type = self.complete_meta_data["APP1:RawThermalImageType"]
        return None if type is None else image_type

    @property
    def get_camera(self) -> [str, None]:
        """ gets the camera used to click the image"""
        return Camera.E50 if Camera.E50 in self.model else Camera.EDGEPRO

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
