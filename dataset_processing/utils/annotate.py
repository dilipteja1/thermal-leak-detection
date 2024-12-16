import os.path
import random

import numpy as np
import matplotlib.pyplot as plt
from dataset_processing.utils.thermogram import Thermal
from dataset_processing.constants import Camera
import math
import cv2

from dataset_processing.utils.ip_algorithms import otsu_thresholding
from dataset_processing.metadata import ImageMetaData

# OUTPUT_PATH = "../../test_data_points/"
TEST_IMAGES = "../../../test_data"

class ThermalIP:
    def __init__(self, training_data_list):
        self.training_list = training_data_list

    def align_visual(self):
        thermal_image = cv2.imread(self.thermal.thermal_image_path)
        visual_image = cv2.imread(self.thermal.visual_image_path)

        thermal_fov_horizontal = Camera.E50_THERMAL_HORIZONTAL_FOV
        thermal_fov_vertical = Camera.E50_THERMAL_VERTICAL_FOV
        visual_fov_horizontal = Camera.E50_OPTICAL_HORIZONTAL_FOV
        visual_fov_vertical = Camera.E50_OPTICAL_VERTICAL_FOV

        visual_height, visual_width, visual_channels = visual_image.shape

        visual_center = {
            'x': math.floor(visual_width / 2),
            'y': math.floor(visual_height / 2)
        }

        thermal_height, thermal_width, thermal_channels = thermal_image.shape

        thermal_center = {
            'x': math.floor(thermal_width / 2),
            'y': math.floor(thermal_height / 2)
        }

        horizontal_scaling = thermal_fov_horizontal / visual_fov_horizontal
        vertical_scaling = thermal_fov_vertical / visual_fov_vertical

        offset_x = math.ceil(visual_center.get('x') * horizontal_scaling)
        offset_y = math.ceil(visual_center.get('y') * vertical_scaling)

        x = visual_center.get('x') - 45
        y = visual_center.get('y') - 45

        cropped_img = visual_image[y - offset_y:y + offset_y, x - offset_x:x + offset_x, :]
        #
        cropped_img = cv2.resize(cropped_img, (640, 480))
        thermal_image_resize = cv2.resize(thermal_image, (640, 480))
        cv2.imshow("cropped visual image", cropped_img)
        cv2.imshow("Resized thermal image", thermal_image_resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @property
    def get_unique_dates(self):
        """ takes in the training list and provides unique dates out of them"""
        return list(set(dp['date_taken'] for dp in self.training_list))

    @property
    def get_unique_leaks(self):
        """ takes in the training list and provides unique dates out of them"""
        return list(set(dp['leak_info'] for dp in self.training_list))

    @property
    def get_unique_windows(self):
        """ takes in the training list and provides unique dates out of them"""
        return list(set(dp['window_name'] for dp in self.training_list))

    def save_thermal_images(self):
        """ Picks the IR image and saves the raw thermal image in RGB color pallete. this will
            remove the watermarks and other fields which might effect image
            processing techniques
        """
        dates = self.get_unique_dates
        leaks = self.get_unique_leaks
        windows = self.get_unique_windows
        # filter the images based on date, opening
        for date in dates:
            for window in windows:
                for leak in leaks:
                    filtered_training_list = list(filter(lambda x: (x['date_taken'] == date and x['leak_info'] == leak
                                                                    and x['window_name'] == window),
                                                         self.training_list))
                    # get the output directory
                    output_dir = f"raw_IR/{date}/{leak}/{window}"
                    if filtered_training_list != [] and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    for dp in filtered_training_list:
                        cm = ImageMetaData(dp['thermal_image_path'])
                        thermal = Thermal(dp['thermal_image_path'], dp['visual_image_path'], cm)
                        render = thermal.thermogram.render_pil()  # Returns Pillow Image object
                        render.save(os.path.join(output_dir, f"{cm.get_image_name}"))
    def save_fahrenheit_images(self):
        """ Picks the IR image and saves the raw thermal image with fahrenheit values.
        """
        dates = self.get_unique_dates
        leaks = self.get_unique_leaks
        windows = self.get_unique_windows
        # filter the images based on date, opening
        for date in dates:
            for window in windows:
                for leak in leaks:
                    filtered_training_list = list(filter(lambda x: (x['date_taken'] == date and x['leak_info'] == leak
                                                                    and x['window_name'] == window),
                                                         self.training_list))
                    # get the output directory
                    output_dir = f"raw_IR_fahrenheit/{date}/{leak}/{window}"
                    if filtered_training_list != [] and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    for dp in filtered_training_list:
                        cm = ImageMetaData(dp['thermal_image_path'])
                        thermal = Thermal(dp['thermal_image_path'], dp['visual_image_path'], cm)
                        render = thermal.thermogram.render_pil()  # Returns Pillow Image object
                        cv2.imwrite(os.path.join(output_dir, f"{cm.get_image_name}"), thermal.thermogram.fahrenheit)

    def thresholding_1(self):
        """ threshold based on filtered images based on data, leak, window"""
        # get unique dates from the training list
        dates = self.get_unique_dates
        leaks = self.get_unique_leaks
        windows = self.get_unique_windows

        # filter the images based on date, opening
        for date in dates:
            for window in windows:
                for leak in leaks:
                    filtered_training_list = list(filter(lambda x: (x['date_taken'] == date and x['leak_info'] == leak
                                                                    and x['window_name'] == window),
                                                         self.training_list))
                    filtered_images = [dp['raw_thermal'] for dp in filtered_training_list]

                    # stack images
                    stacked_image = None
                    for img in filtered_images:
                        if stacked_image is None:
                            stacked_image = img
                        else:
                            stacked_image = np.concatenate((stacked_image, img), axis=1)

                    if stacked_image is None:
                        print("No images with given date and leak parameters")
                        continue

                    # get the output directory
                    output_dir = f"thresholding_1/{date}/{leak}/{window}"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)

                    # threshold
                    stacked_image = cv2.cvtColor(stacked_image, cv2.COLOR_BGR2GRAY)
                    blurred_image = cv2.GaussianBlur(stacked_image, (5, 5), 0)
                    threshold, thresholded_image = cv2.threshold(blurred_image, 0, 255,
                                                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    print(f"the Threshold found by otsu for {window} window is {threshold}")
                    # get the histogram of pixel values vs counts
                    plt.clf()
                    plt.hist(stacked_image.ravel(), bins=255, range=(0, 255), color='blue')
                    plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold', linewidth=1)
                    plt.title('Pixel Count Histogram')
                    plt.xlabel('Pixel Intensity')
                    plt.ylabel('Count')
                    plt.savefig(os.path.join(output_dir, 'histogram.jpg'))

                    # threshold all the images in the directory
                    for dp in filtered_training_list:
                        cm = ImageMetaData(dp['thermal_image_path'])
                        thermal = Thermal(dp['thermal_image_path'], dp['visual_image_path'], cm)
                        original_image = dp['raw_thermal']
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                        blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)
                        _, thresholded_image = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY_INV)
                        combined_image = np.hstack((original_image, thresholded_image))
                        # save
                        cv2.imwrite(os.path.join(output_dir, f"{cm.get_image_name}"), combined_image)

        print("Thresholding successful")

    def thresholding_2(self):
        """ threshold all the images individually"""
        # get unique dates from the training list
        dates = self.get_unique_dates
        leaks = self.get_unique_leaks
        windows = self.get_unique_windows

        # filter the images based on date, opening
        for date in dates:
            for window in windows:
                for leak in leaks:
                    filtered_training_list = list(filter(lambda x: (x['date_taken'] == date and x['leak_info'] == leak
                                                                    and x['window_name'] == window),
                                                         self.training_list))
                    filtered_images = [dp['raw_thermal'] for dp in filtered_training_list]

                    # get the output directory
                    output_dir = f"thresholding_2/{date}/{leak}/{window}"
                    if filtered_images != [] and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)

                    for image in filtered_training_list:
                        # threshold
                        grayscale_image = cv2.cvtColor(image['raw_thermal'], cv2.COLOR_BGR2GRAY)
                        blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
                        threshold, thresholded_image = cv2.threshold(blurred_image, 0, 255,
                                                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        print(f"the Threshold found by otsu for {window} window is {threshold}")
                        # get the histogram of pixel values vs counts
                        plt.clf()
                        plt.hist(image['raw_thermal'].ravel(), bins=255, range=(0, 255), color='blue')
                        plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold', linewidth=1)
                        plt.title('Temperature Count Histogram')
                        plt.xlabel('Temperature Intensity')
                        plt.ylabel('Count')
                        plt.savefig(os.path.join(output_dir, 'histogram.jpg'))

                        cm = ImageMetaData(image['thermal_image_path'])
                        combined_image = np.hstack((grayscale_image, thresholded_image))
                        # save
                        cv2.imwrite(os.path.join(output_dir, f"{cm.get_image_name}.jpg"), combined_image)

        print("Thresholding successful")

    def thresholding_3(self):
        """ threshold all the images filtered based on date window"""
        # get unique dates from the training list
        dates = self.get_unique_dates
        leaks = self.get_unique_leaks
        windows = self.get_unique_windows

        # filter the images based on date, opening
        for date in dates:
            for window in windows:
                filtered_training_list = list(filter(lambda x: (x['date_taken'] == date
                                                                and x['window_name'] == window), self.training_list))
                filtered_images = [dp['raw_thermal'] for dp in filtered_training_list]

                # stack images
                stacked_image = None
                for img in filtered_images:
                    if stacked_image is None:
                        stacked_image = img
                    else:
                        stacked_image = np.concatenate((stacked_image, img), axis=1)

                if stacked_image is None:
                    print("No images with given date and leak parameters")
                    continue
                # get the output directory
                output_dir = f"thresholding_3/{date}/{window}"
                if filtered_images != [] and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                # threshold
                stacked_image = cv2.cvtColor(stacked_image, cv2.COLOR_BGR2GRAY)
                blurred_image = cv2.GaussianBlur(stacked_image, (5, 5), 0)
                threshold, thresholded_image = cv2.threshold(blurred_image, 0, 255,
                                                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                print(f"the Threshold found by otsu for {window} window is {threshold}")

                # get the histogram of pixel values vs counts
                plt.clf()
                plt.hist(stacked_image.ravel(), bins=255, range=(0, 255), color='blue')
                plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold', linewidth=1)
                plt.title('Pixel Count Histogram')
                plt.xlabel('Pixel Intensity')
                plt.ylabel('Count')
                plt.savefig(os.path.join(output_dir, 'histogram.jpg'))

                # threshold all the images in the directory
                for dp in filtered_training_list:
                    cm = ImageMetaData(dp['thermal_image_path'])
                    thermal = Thermal(dp['thermal_image_path'], dp['visual_image_path'], cm)
                    original_image = dp['raw_thermal']
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                    blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)
                    _, thresholded_image = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY_INV)
                    combined_image = np.hstack((original_image, thresholded_image))
                    # save
                    cv2.imwrite(os.path.join(output_dir, f"{cm.get_image_name}"), combined_image)

        print("Thresholding successful")

    def thresholding_4(self):
        """ threshold all the images filtered based on date window"""
        # get unique dates from the training list
        dates = self.get_unique_dates
        leaks = self.get_unique_leaks
        windows = self.get_unique_windows

        # filter the images based on date, opening
        for date in dates:
            for window in windows:
                filtered_training_list = list(filter(lambda x: (x['date_taken'] == date
                                                                and x['window_name'] == window), self.training_list))
                filtered_images = [dp['fahrenheit'].astype(np.uint8) for dp in filtered_training_list]

                # stack images
                stacked_image = None
                for img in filtered_images:
                    if stacked_image is None:
                        stacked_image = img
                    else:
                        stacked_image = np.concatenate((stacked_image, img), axis=1)

                if stacked_image is None:
                    print("No images with given date and leak parameters")
                    continue
                # get the output directory
                output_dir = f"thresholding_4/{date}/{window}"
                if filtered_images != [] and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                # threshold
                # stacked_image = cv2.cvtColor(stacked_image, cv2.COLOR_BGR2GRAY)
                blurred_image = cv2.GaussianBlur(stacked_image, (5, 5), 0)
                threshold, thresholded_image = cv2.threshold(blurred_image, 0, 255,
                                                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                print(f"the Threshold found by otsu for {window} window is {threshold}")

                # get the histogram of pixel values vs counts
                plt.clf()
                plt.hist(stacked_image.ravel(), bins=100, range=(0, 100), color='blue')
                plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold', linewidth=1)
                plt.title('Pixel Count Histogram')
                plt.xlabel('Pixel Intensity')
                plt.ylabel('Count')
                plt.savefig(os.path.join(output_dir, 'histogram.jpg'))

                # threshold all the images in the directory
                for dp in filtered_training_list:
                    cm = ImageMetaData(dp['thermal_image_path'])
                    thermal = Thermal(dp['thermal_image_path'], dp['visual_image_path'], cm)
                    original_image = dp['fahrenheit']
                    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                    blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)
                    _, thresholded_image = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY_INV)
                    combined_image = np.hstack((original_image, thresholded_image))
                    # save
                    cv2.imwrite(os.path.join(output_dir, f"{cm.get_image_name}"), combined_image)

        print("Thresholding successful")

    def thresholding_5(self):
        """ threshold based on filtered images based on data, leak, window . fahrenheit values"""
        # get unique dates from the training list
        dates = self.get_unique_dates
        leaks = self.get_unique_leaks
        windows = self.get_unique_windows

        # filter the images based on date, opening
        for date in dates:
            for window in windows:
                for leak in leaks:
                    filtered_training_list = list(filter(lambda x: (x['date_taken'] == date and x['leak_info'] == leak
                                                                    and x['window_name'] == window),
                                                         self.training_list))
                    filtered_images = [dp['fahrenheit'].astype(np.uint8) for dp in filtered_training_list]

                    # stack images
                    stacked_image = None
                    for img in filtered_images:
                        if stacked_image is None:
                            stacked_image = img
                        else:
                            stacked_image = np.concatenate((stacked_image, img), axis=1)

                    if stacked_image is None:
                        print("No images with given date and leak parameters")
                        continue

                    # get the output directory
                    output_dir = f"thresholding_5/{date}/{leak}/{window}"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)

                    # threshold
                    blurred_image = cv2.GaussianBlur(stacked_image, (5, 5), 0)
                    threshold, thresholded_image = cv2.threshold(blurred_image, 0, 255,
                                                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    print(f"the Threshold found by otsu for {window} window is {threshold}")
                    # get the histogram of pixel values vs counts
                    plt.clf()
                    plt.hist(stacked_image.ravel(), bins=100, range=(0, 100), color='blue')
                    plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold', linewidth=1)
                    plt.title('Pixel Count Histogram')
                    plt.xlabel('Pixel Intensity')
                    plt.ylabel('Count')
                    plt.savefig(os.path.join(output_dir, 'histogram.jpg'))

                    # threshold all the images in the directory
                    for dp in filtered_training_list:
                        cm = ImageMetaData(dp['thermal_image_path'])
                        thermal = Thermal(dp['thermal_image_path'], dp['visual_image_path'], cm)
                        original_image = dp['fahrenheit']
                        blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)
                        _, thresholded_image = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY_INV)
                        combined_image = np.hstack((original_image, thresholded_image))
                        # save
                        cv2.imwrite(os.path.join(output_dir, f"{cm.get_image_name}"), combined_image)

        print("Thresholding successful")

    def thresholding_6(self):
        """ threshold all the images individually"""
        # get unique dates from the training list
        dates = self.get_unique_dates
        leaks = self.get_unique_leaks
        windows = self.get_unique_windows

        # filter the images based on date, opening
        for date in dates:
            for window in windows:
                for leak in leaks:
                    filtered_training_list = list(filter(lambda x: (x['date_taken'] == date and x['leak_info'] == leak
                                                                    and x['window_name'] == window),
                                                         self.training_list))
                    filtered_images = [dp['fahrenheit'].astype(np.uint8) for dp in filtered_training_list]

                    # get the output directory
                    output_dir = f"thresholding_6/{date}/{leak}/{window}"
                    if filtered_images != [] and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)

                    for dp in filtered_training_list:
                        # threshold
                        image = dp['fahrenheit'].astype(np.uint8)
                        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
                        threshold, thresholded_image = cv2.threshold(blurred_image, 0, 255,
                                                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        print(f"the Threshold found by otsu for {window} window is {threshold}")
                        # get the histogram of pixel values vs counts
                        plt.clf()
                        plt.hist(image.ravel(), bins=100, range=(0, 100), color='blue')
                        plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold', linewidth=1)
                        plt.title('Temperature Count Histogram')
                        plt.xlabel('Temperature Intensity')
                        plt.ylabel('Count')
                        plt.savefig(os.path.join(output_dir, 'histogram.jpg'))

                        cm = ImageMetaData(dp['thermal_image_path'])
                        combined_image = np.hstack((image, thresholded_image))
                        # save
                        cv2.imwrite(os.path.join(output_dir, f"{cm.get_image_name}.jpg"), combined_image)

        print("Thresholding successful")

    def manual_thresholding(self):
        """ Manually provide the threshold and see the segmentation results"""
