import cv2
import numpy as np


def create_mask(image, lower_bgr, upper_bgr):
    """
    Create a binary mask using a BGR color range.

    Parameters:
        image (numpy.ndarray): Input BGR image.
        lower_bgr (tuple/list): Lower BGR bound.
        upper_bgr (tuple/list): Upper BGR bound.

    Returns:
        numpy.ndarray: Binary mask.
    """

    lower = np.array(lower_bgr, dtype=np.uint8)
    upper = np.array(upper_bgr, dtype=np.uint8)

    mask = cv2.inRange(
        image,
        lower,
        upper
    )

    print("[INFO] Binary mask created.")

    return mask


def apply_mask(image, mask):
    """
    Apply a binary mask to an image.

    Parameters:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Binary mask.

    Returns:
        numpy.ndarray: Masked image.
    """

    result = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    print("[INFO] Mask applied.")

    return result
