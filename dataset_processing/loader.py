import argparse
import os
from pathlib import Path
import cv2
from collections import defaultdict
from dataset_processing.utils.annotate import ThermalIP
from metadata import ImageMetaData
from utils.thermogram import Thermal
from constants import Camera

DATASET_PATH = "../test_data"
TEST_DATA_PATH = "../test_data_points"
PROCESSED_TEST_DATA_PATH = "../test_data_processed"

class RawDataLoader:
    def __init__(self, input_path=DATASET_PATH, output_path=PROCESSED_TEST_DATA_PATH):
        # paths
        self.input_path = input_path
        self.output_path = output_path
        self.thermal_image_file = None  # path of the thermal image
        self.data_point = defaultdict(str)
        self.cm = [ImageMetaData, None]
        self.thermal = [Thermal, None]
        self.camera = [str, None]

    def load(self):
        """
             takes the raw dataset directory as input and provides processed out put
                Returns:
                        None
        """
        # check if the path exists
        if not os.path.isdir(self.input_dir):
            print("the input directory provided is invalid please provide a valid one")
        for folder_name, sub_folders, filenames in os.walk(self.input_dir):
            for filename in filenames:
                if filename.endswith(('.jpg', '.jpeg')):
                    image_file = os.path.join(folder_name, filename)
                    self.cm = ImageMetaData(image_file)
                    if self.cm.image_type.lower() == Camera.THERMAL_IMAGE_TYPE:
                        self.thermal_image_file = image_file
                        self.thermal = Thermal(self.thermal_image_file, self.get_visual_image_file(), self.cm)
                        self.pre_annotate()

    def get_visual_image_file(self):
        """
            Picks an IR image file and get the corresponding DC image file
            and visual image
            #TODO make a diverse data point (text + images)
        Returns:
            data point tuple
        """
        if self.cm.get_camera == Camera.E50:
            folder_name = os.path.dirname(self.thermal_image_file)
            file_name, extension = os.path.splitext(os.path.basename(self.thermal_image_file))
            thermal_image_number = int(file_name.split('_')[1])
            visual_image_number = thermal_image_number + 1
            visual_image_name = "DC_" + str(visual_image_number) + ".jpg"
            visual_image_file = os.path.join(folder_name, visual_image_name)
            if os.path.exists(visual_image_file):
                print(f"Visual Image found: {visual_image_name}")
                return Path(visual_image_file)
            else:
                print(
                    f"visual image for the corresponding Thermal image {os.path.basename(self.thermal_image_file)} not found")
                return None
    def pre_annotate(self):
        """
            performs bunchof image processing techniques so that the resulting image is ready for
            annotation
        Returns:
            data point with image processed information
        """
        thermal_processor = ThermalIP(self.thermal)
        if self.cm.get_camera == Camera.E50:
            alinged_visual = thermal_processor.align_visual()
            self.thermal.visual = alinged_visual
        palettes = ["turbo", "cividis", "inferno", "grayscale", "hot"]
        for p in palettes:
            # The below call returns a Pillow Image object.
            # A sibling method called `render` returns a numpy array.
            render = self.thermal.thermogram.render_pil(
                min_v=40,
                max_v=50,
                unit="fahrenheit",
                palette=p,
            )
            render.save(f"render-{p}.png")
            # mask = self.thermal.thermogram.fahrenheit > self.thermal.thermogram.fahrenheit.mean()
            # self.thermal.thermogram.picture_in_picture_pil(mask=mask, mask_mode="classical").save("pip_classical.png")
            # self.thermal.thermogram.picture_in_picture_pil(mask=mask, mask_mode="alternative").save("pip_alternative.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data set loader for pre processing")
    parser.add_argument("--input_dir", "--input-dir", help="the dataset directory to be processed")
    parser.add_argument("--output_dir", "--output-dir", help="output directory")

    args = parser.parse_args()
    loader = RawDataLoader()
    loader.load()
