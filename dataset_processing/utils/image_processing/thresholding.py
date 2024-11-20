import cv2
import matplotlib.pyplot as plt

def otsu_thresholding(thermal_image):
    """
        method to perform otsu thresholding on the thermal image
    Returns:
        a binary image which specifies two different classes
    """

    thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise before thresholding
    blurred_image = cv2.GaussianBlur(thermal_image, (5, 5), 0)

    # Apply Otsu's thresholding
    # cv2.THRESH_BINARY applies binary thresholding
    # cv2.THRESH_OTSU flag automatically determines the threshold value
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display the original and threshold images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(thermal_image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Otsu Thresholding")
    plt.imshow(thresholded_image, cmap='gray')
    plt.show()

    return thresholded_image

