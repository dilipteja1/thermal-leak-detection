import argparse
import os
import json
from pathlib import Path
import pandas as pd
from datetime import datetime as dt
from dataset_processing.utils.annotate import ThermalIP
from metadata import ImageMetaData
from utils.thermogram import Thermal
from constants import Camera

DATASET_PATH = "../test_data"
TEST_DATA_PATH = "../test_data_points"
PROCESSED_DATASET_PATH = "../test_data_processed"
DATASET_JSON = "dataset.json"
EXCEL_DATA_PATH = "../thermal_collection.xlsx"


class DataReader:
    def __init__(self, input_path=DATASET_PATH, output_path=PROCESSED_DATASET_PATH):
        # paths
        self.input_path = input_path
        self.output_path = output_path
        self.cm = [ImageMetaData, None]
        self.thermal = [Thermal, None]
        self.camera = [str, None]
        self.training_list = []
        self.testing_list = []

        # excel data
        self.df_collection = None
        self.df_windows = None

    def read_json(self):
        """
            takes the json file of dataset and reads the data
        :return:
            dictionary of the dataset. with list of data points
        """
        with open(DATASET_JSON, 'r') as f:
            data = json.load(f)
            self.training_list = data['train']
            self.testing_list = data['test']
            if data is None:
                self.load()

    def save_json(self):
        """ save the training data into json for faster retrieval"""
        with open(DATASET_JSON, "w") as outfile:
            json.dump({'train': self.training_list, 'test': self.testing_list}, outfile)

    def read_excel(self):
        """
            Reads the excel data collection and windows information into dataframes
        :return:
            pandas dataframe
        """
        self.df_collection = pd.read_excel(EXCEL_DATA_PATH, sheet_name='Collection')
        self.df_windows = pd.read_excel(EXCEL_DATA_PATH, sheet_name='Windows')

    def load(self):
        """
            takes the raw dataset directory as input and starts generating datapoints
            Returns:
                    None
        """
        if os.path.exists(DATASET_JSON):
            self.read_json()
            for index in range(len(self.training_list)):
                dp = self.training_list[index]
                self.training_list[index] = self.expand_data_point(dp)
            for index in range(len(self.testing_list)):
                dp = self.testing_list[index]
                self.testing_list[index] = self.expand_data_point(dp)
            return

        if not os.path.exists(EXCEL_DATA_PATH):
            print("Excel data sheet not found.")
            return
        else:
            self.read_excel()

        for folder_name, sub_folders, filenames in os.walk(self.input_path):
            for filename in filenames:
                if filename.__contains__("IR"):
                    image = os.path.join(folder_name, filename)
                    self.training_list.append(self.create_data_point(image))
        self.save_json()

    def expand_data_point(self, dp):
        """takes json data point and generates numpy arrays into it
            :return
                dp with more
        """
        self.cm = ImageMetaData(dp['thermal_image_path'])
        thermogram =  thermogram = Thermal(dp['thermal_image_path'], dp['visual_image_path'], self.cm)
        dp['raw_thermal'] = thermogram.get_thermal
        dp['raw_visual'] = thermogram.get_visual
        return dp
    def create_data_point(self, image):
        """
            takes in thermal image file name and folder names and creates a data point
        :return:
            dictionary
        """
        dp = {}
        dp['thermal_image_path'] = image
        self.cm = ImageMetaData(dp['thermal_image_path'])
        dp['visual_image_path'] = str(self.get_visual_image_file())


        # todo to call the aligner
        dp['aligned_visual'] = ""
        dp['date_taken'] = self.cm.get_date
        # todo get the indoor and outdoor temp from the methods
        dp['indoor_temperature'] = 68
        dp['outdoor_temperature'] = 64
        dp['leak_info'] = self.cm.get_opening
        dp['mask'] = None
        dp['window_info'] = None
        return dp

    def get_indoor_temperature(self):
        """
            gets the indoor temperature based on house and date
        :return:
            indoor temperature value
        """
        return self.df_collection.loc[
            (self.df_collection['House'] == 1 and dt(self.df_collection['Date']).date() == dt(self.cm.get_date).date()),'TempFInsideStart']

    def get_outdoor_temperature(self):
        """
            gets the outdoor temperature based on house and date
        :return:
            outdoor temperture value
        """
        #todo get the outdoor temperature values from Weather.com
        return self.df_collection.loc[
            (self.df_collection['House'] == 1 & self.df_collection['Date'] == self.cm.get_date.date()), 'TempFOutsideStart']

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
        thermal_ip = ThermalIP(self.training_list)
        thermal_ip.thresholding_wo_normalization()
        # if self.cm.get_camera == Camera.E50:
        #     alinged_visual = thermal_processor.align_visual()
        #     self.thermal.visual = alinged_visual
        # palettes = ["turbo", "cividis", "inferno", "grayscale", "hot"]
        # for p in palettes:
        #     # The below call returns a Pillow Image object.
        #     # A sibling method called `render` returns a numpy array.
        #     render = self.thermal.thermogram.render_pil(
        #         min_v=40,
        #         max_v=50,
        #         unit="fahrenheit",
        #         palette=p,
        #     )
        #     render.save(f"render-{p}.png")
            # mask = self.thermal.thermogram.fahrenheit > self.thermal.thermogram.fahrenheit.mean()
            # self.thermal.thermogram.picture_in_picture_pil(mask=mask, mask_mode="classical").save("pip_classical.png")
            # self.thermal.thermogram.picture_in_picture_pil(mask=mask, mask_mode="alternative").save("pip_alternative.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data set loader for pre processing")
    parser.add_argument("--input_dir", "--input-dir", help="the dataset directory to be processed")
    parser.add_argument("--output_dir", "--output-dir", help="output directory")

    args = parser.parse_args()
    reader = DataReader(input_path=DATASET_PATH, output_path=PROCESSED_DATASET_PATH)
    reader.load()
    reader.pre_annotate()
