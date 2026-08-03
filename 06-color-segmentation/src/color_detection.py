import cv2
import numpy as np


def detect_color(image, lower_hsv, upper_hsv):
    """
    Detect a specific color in an image.

    Parameters:
        image (numpy.ndarray): Input BGR image.
        lower_hsv (tuple/list): Lower HSV bound.
        upper_hsv (tuple/list): Upper HSV bound.

    Returns:
        tuple:
            mask - Binary mask
            result - Detected color image
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    result = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    pixels = cv2.countNonZero(mask)

    print(f"[INFO] Detected pixels: {pixels}")

    return mask, result
