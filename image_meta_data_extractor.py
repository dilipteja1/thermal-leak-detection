"""
Title : Thermal data set processing file
Author : Dilip teja
"""

import os
import argparse
import json
import tomllib
import exiftool
from openpyxl import Workbook
from pathlib import Path


class FlirMetaDataExtractor:

    def __init__(self, toml_file=None):
        self.toml = self.load_toml(toml_file)
        self.environment = ""
        self.sun_direction = ""
        self.image_tags = ""
        self.images_dir = ""
        self.window_id = ""

    @staticmethod
    def load_toml(toml_file):
        """Load TOML data from file"""
        try:
            with open(toml_file, "rb") as f:
                toml_data: dict = tomllib.load(f)
            return toml_data
        except Exception as e:
            print("Error while loading the toml. " + str(e))
            exit()

    def get_image_meta_data(self):
        """
        Start the processing of the thermal images in the directory
        Returns:
            filtered list of all the metadata of all images present in the folder
        """
        image_data = []
        try:
            self.images_dir = Path(self.toml["data_path"]["path"])
            if self.images_dir.exists():
                print("data Path found: " + str(self.images_dir))
            else:
                print("Incorrect/Invalid data path")
            self.image_tags = self.toml["image_tags"]
            image_file_tags = self.image_tags["file"].values()
            image_app1_tags = self.image_tags["app1"].values()
            tags = list(image_file_tags) + list(image_app1_tags)
            with exiftool.ExifToolHelper() as et:
                meta_data = et.get_tags(self.images_dir, tags)

            for each_image_data in meta_data:
                image_data.append({tag.split(":")[1]: each_image_data[tag] for tag in each_image_data if
                                   tag.startswith("APP1") or tag.startswith("File")})
            return image_data
        except FileNotFoundError as e:
            print(str(e), "Make sure the path provided is correct")
        except Exception as e:
                print("Error while executing exiftool: Please check exiftool installation " \
                      "and make sure the data folder path is valid and have images.\nError: " + str(e))
                exit()

    def make_data_sheet(self, images_data):
        """
        function to build the excel sheet of all the relevant information
        Returns:
            saves all the meta information about an image in an excel sheet
        """
        try:
            building_data = self.toml["building"]
            room_data = self.toml["room"]
            environment_data = self.toml["environment"]
            window_data = self.toml["window"]

            data_point = {}
            # Create a workbook
            wb = Workbook()
            # select the active worksheet
            ws = wb.active
            ws.title = "window_" + self.toml["data_path"]["id"]
            # to handle single image scenario

            heading = (images_data[0] | environment_data | building_data | room_data | window_data).keys()
            ws.append(list(heading))
            # image_info
            for each_image in images_data:
                data_point = (each_image | environment_data | building_data | room_data | window_data).values()
                ws.append(list(data_point))

            wb.save("window_" + self.toml["data_path"]["id"] + ".xlsx")
            print("Data sheet creation successful")
        except PermissionError as e:
            print(str(e), ". Please close the workbook to save the changes.")
        except Exception as e:
            print("Unexpected Error occured: " + str(e))
            exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flir Thermal image metadata extractor")
    parser.add_argument("-toml", "--toml", help="toml file which contains meta data to be collected across the images")

    args = parser.parse_args()
    metadata_extractor = FlirMetaDataExtractor(toml_file=args.toml)
    image_data = metadata_extractor.get_image_meta_data()
    metadata_extractor.make_data_sheet(image_data)
