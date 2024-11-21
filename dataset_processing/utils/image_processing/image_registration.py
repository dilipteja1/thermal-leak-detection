from __future__ import print_function
import numpy as np
import cv2
import os
import math

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def register4(thermal_image_file, visual_image_file):
    """
        registers the thermal image onto the visual image. it is similar to aligning the images
        to match the point of views
    Args:
        thermal_image:
        visual_image:

    Returns:
        thermal_image and visual_image (aligned)
    """
    thermal_image = cv2.imread(thermal_image_file)
    visual_image = cv2.imread(visual_image_file)

    thermal_image_resized = cv2.resize(thermal_image, (visual_image.shape[1], visual_image.shape[0]))

    # Convert images to grayscale for feature detection
    gray_thermal = cv2.cvtColor(thermal_image_resized, cv2.COLOR_BGR2GRAY)
    gray_visual = cv2.cvtColor(visual_image, cv2.COLOR_BGR2GRAY)

    # Use ORB detector to find keypoints and descriptors
    orb = cv2.ORB_create(500)
    keypoints_thermal, descriptors_thermal = orb.detectAndCompute(gray_thermal, None)
    keypoints_visual, descriptors_visual = orb.detectAndCompute(gray_visual, None)

    # Use BFMatcher to find matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_thermal, descriptors_visual)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points_thermal = np.float32([keypoints_thermal[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points_visual = np.float32([keypoints_visual[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography matrix
    H, _ = cv2.findHomography(points_thermal, points_visual, cv2.RANSAC, 5.0)

    # Warp thermal image using the homography matrix
    aligned_thermal_image = cv2.warpPerspective(thermal_image_resized, H,
                                                (visual_image.shape[1], visual_image.shape[0]))

    # Optionally, overlay images to check alignment
    # overlay = cv2.addWeighted(visual_image, 0.5, aligned_thermal_image, 0.5, 0)

    # Display the aligned thermal image and overlay
    cv2.imshow('Aligned Thermal Image', aligned_thermal_image)
    # cv2.imshow('Overlay', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the aligned image
    cv2.imwrite('aligned_thermal_image.jpg', aligned_thermal_image)
    # cv2.imwrite('overlay.jpg', overlay)


def register2(thermal_image_file, visual_image_file):
    thermal_image = cv2.imread(thermal_image_file)
    visual_image = cv2.imread(visual_image_file)

    # visual_image = cv2.resize(visual_image, (thermal_image.shape[1], thermal_image.shape[0]))

    thermal_fov_horizontal = 25
    thermal_fov_vertical = 19
    visual_fov_horizontal = 53
    visual_fov_vertical = 41
    # cv2.imshow("original Visual Image", visual_image)

    scaling_factor_horizontal = thermal_fov_horizontal / visual_fov_horizontal
    scaling_factor_vertical = thermal_fov_vertical / visual_fov_vertical

    new_width = int(visual_image.shape[1] * scaling_factor_horizontal)
    new_height = int(visual_image.shape[0] * scaling_factor_vertical)

    resized_visual_image = cv2.resize(visual_image, (new_width, new_height))

    if resized_visual_image.shape != thermal_image.shape:
        thermal_image = cv2.resize(thermal_image, (resized_visual_image.shape[1], resized_visual_image.shape[0]))

    cv2.imshow('Resized_visual_image', resized_visual_image)
    cv2.imshow('thermal_image', thermal_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



