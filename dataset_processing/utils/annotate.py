import os.path
import random

import numpy as np
import matplotlib.pyplot as plt
from dataset_processing.utils.thermogram import Thermal
from dataset_processing.constants import Camera
import math
import cv2

from dataset_processing.utils.ip_algorithms import otsu_thresholding

OUTPUT_PATH = "../../test_data_points/"


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

    def thresholding_wo_normalization(self):
        """ provide isotherm based on the temperature value"""
        dates = []
        filtered_training_list = []
        for dp in self.training_list:
            date = dp['date_taken']
            if date not in dates:
                dates.append(date)
            if date == dates[0]:
                filtered_training_list.append(dp)

        hstack_image = None
        for dp in filtered_training_list:
            if hstack_image is None:
                img = dp['raw_thermal']
                hstack_image = img
            else:
                img = dp['raw_thermal']
                hstack_image = np.concatenate((hstack_image, img), axis=1)

        global_threshold, thresholded_hstack_image = otsu_thresholding(hstack_image, display=True)

        random_dp_indices = random.sample(range(1, len(filtered_training_list)), 10)
        output_dir = "thresholding_wo_normalization"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        for index in random_dp_indices:
            original_image = filtered_training_list[index]['raw_thermal']
            _, thresholded_image = cv2.threshold(original_image, global_threshold, 255, cv2.THRESH_BINARY)
            combined_image = np.hstack((original_image, thresholded_image))
            # save
            cv2.imwrite(os.path.join(output_dir, f'{index}.jpg'), combined_image)
