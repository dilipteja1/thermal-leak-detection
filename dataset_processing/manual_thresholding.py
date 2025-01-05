from flyr import unpack
import os
import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
DATA = "../test_data_points"
Adaptive = "../test_data_processed/adaptive"
Gradient = "../test_data_processed/gradient"


def all_temp_threshold():
    """ every pixel value act as a threshold and get seperated"""

    for image in os.listdir(DATA):
        image_name = image.split('.')[0]
        image_path = os.path.join(DATA, image)
        thermal = unpack(image_path)
        output_dir = f"{image_name}"
        if not os.path.exists(image_name):
            os.makedirs(output_dir, exist_ok=True)
        fahrenheit = thermal.fahrenheit
        print(fahrenheit.shape)
        min_temp = int(np.min(fahrenheit))
        max_temp = int(np.max(fahrenheit))
        for temp in range(min_temp + 1, max_temp + 1):
            # The below call returns a Pillow Image object.
            # A sibling method called `render` returns a numpy array.
            render = thermal.render_pil(
                min_v=min_temp,
                max_v=temp,
                unit="fahrenheit",
                palette="grayscale",
            )
            render.save(os.path.join(output_dir, f"{temp}.png"))

def adaptive_thresholding():
        for image in os.listdir(DATA):
            image_name = image.split('.')[0]
            image_path = os.path.join(DATA, image)
            thermal = unpack(image_path)
            render = thermal.render_pil(
                 unit="fahrenheit",
                 palette="grayscale",
            )
            render.save(os.path.join(Adaptive, f"{image_name}.png"))


        for image in os.listdir(Adaptive):
            img = cv.imread(os.path.join(Adaptive,image), cv.IMREAD_GRAYSCALE)
            assert img is not None, "file could not be read, check with os.path.exists()"
            img = cv.medianBlur(img,5)
            ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
            th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
            th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
            cv.imwrite(os.path.join(Adaptive, f'adaptive-{image_name}-mean.jpg'), th2)
            cv.imwrite(os.path.join(Adaptive, f'adaptive-{image_name}-gaussian.jpg'), th3)

def gradient():
    for image in os.listdir(DATA):
        image_name = image.split('.')[0]
        image_path = os.path.join(DATA, image)
        thermal = unpack(image_path)
        render = thermal.render_pil(
            unit="fahrenheit",
            palette="grayscale",
        )
        render.save(os.path.join(Gradient, f"{image_name}.png"))

    for image in os.listdir(Gradient):
        img = cv.imread(os.path.join(Gradient, image), cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"

        # Compute gradients along the X and Y axes using Sobel operator
        grad_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude and direction
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient_direction = np.arctan2(grad_y, grad_x)

        # Normalize the gradient magnitude for visualization
        normalized_magnitude = cv.normalize(gradient_magnitude, None, 0, 255, cv.NORM_MINMAX)
        cv.imwrite(os.path.join(Gradient, f'gradient-{image_name}.jpg'), normalized_magnitude)


# adaptive_thresholding()
# all_temp_threshold()
