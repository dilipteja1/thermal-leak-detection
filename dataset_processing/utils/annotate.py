from dataset_processing.utils.thermogram import Thermal
from dataset_processing.constants import Camera
import math
import cv2


class ThermalIP:
    def __init__(self, thermal: [Thermal]):
        self.thermal = thermal

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


    def thresholding(self):
        """ provide isotherm based on the temperature value"""
        temp_value = 55
        self.thermal

