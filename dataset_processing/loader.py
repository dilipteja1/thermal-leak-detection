import argparse
import os
import json
from ctypes import Union
from pathlib import Path
import pandas as pd
from datetime import datetime as dt
from dataset_processing.utils.annotate import ThermalIP
from datapoint import DataPoint
import cv2

DATASET_PATH = "../test_data"
TEST_DATA_PATH = "../test_data_points"
PROCESSED_DATASET_PATH = "../test_data_processed"
DATASET_JSON = "dataset.json"
EXCEL_DATA_PATH = "../thermal_collection.xlsx"


class DataLoader:
    def __init__(self, input_path=DATASET_PATH, output_path=PROCESSED_DATASET_PATH):
        # paths
        self.input_path = input_path
        self.output_path = output_path
        self.training_list = []
        self.testing_list = []
        self.data_point: Union[DataPoint, None]

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

        # walking through the data from E50
        for folder_name, sub_folders, filenames in os.walk(self.input_path):
            for filename in filenames:
                if filename.__contains__("IR"):
                    dp = {}
                    thermal_image_path = os.path.join(folder_name, filename)
                    dp['thermal_image_path'] = thermal_image_path
                    self.training_list.append(dp)
        self.save_json()
        self.load()

    def expand_data_point(self, dp):
        """takes json data point and generates numpy arrays into it
            :return
                dp with more attributes
        """
        self.data_point = DataPoint(dp['thermal_image_path'])

        dp['raw_thermal'] = self.data_point.get_thermal  # grayscale palette thermal image
        dp['raw_visual'] = self.data_point.get_visual
        dp['thermal_grayscale'] = cv2.cvtColor(dp['raw_thermal'], cv2.COLOR_BGR2GRAY)
        dp['fahrenheit'] = self.data_point.thermogram.fahrenheit
        # todo to call the aligner
        dp['aligned_visual'] = ""
        dp['date_taken'] = self.data_point.get_date
        # todo get the indoor and outdoor temp from the methods
        dp['indoor_temperature'] = 68
        dp['outdoor_temperature'] = 64
        dp['leak_info'] = self.data_point.get_opening
        dp['mask'] = None
        dp['window_name'] = self.data_point.get_window

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

    def annotate(self):
        """
            performs bunchof image processing techniques so that the resulting image is ready for
            annotation
        Returns:
            data point with image processed information
        """
        thermal_ip = ThermalIP(self.training_list)
        # thermal_ip.save_thermal_images()
        # thermal_ip.thresholding_4()
        thermal_ip.manual_thresholding_batch2()
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
    reader = DataLoader(input_path=DATASET_PATH, output_path=PROCESSED_DATASET_PATH)
    reader.load()
    reader.annotate()
