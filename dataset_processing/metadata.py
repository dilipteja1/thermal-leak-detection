from flyr.camera_metadata import CameraMetadata
from dataset_processing.constants import Camera
import exiftool
from collections import defaultdict

class ImageMetaData(CameraMetadata):
    def __init__(self, image):
        super().__init__(image)
        self.image = image

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
