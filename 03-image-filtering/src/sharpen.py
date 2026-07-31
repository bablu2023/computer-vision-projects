import cv2
import numpy as np


def sharpen_image(image):
    """
    Sharpen an image using a convolution kernel.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Sharpened image.
    """

    # Sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel
    )

    print("[INFO] Image sharpened successfully.")

    return sharpened
