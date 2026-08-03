import cv2
import numpy as np


def scharr_edge_detection(gray):
    """
    Apply Scharr edge detection.

    Parameters:
        gray (numpy.ndarray): Grayscale image.

    Returns:
        tuple:
            scharr_x
            scharr_y
            scharr_magnitude
    """

    scharr_x = cv2.Scharr(
        gray,
        cv2.CV_64F,
        1,
        0
    )

    scharr_y = cv2.Scharr(
        gray,
        cv2.CV_64F,
        0,
        1
    )

    abs_x = cv2.convertScaleAbs(
        scharr_x
    )

    abs_y = cv2.convertScaleAbs(
        scharr_y
    )

    magnitude = cv2.magnitude(
        scharr_x,
        scharr_y
    )

    magnitude = cv2.convertScaleAbs(
        magnitude
    )

    print("[INFO] Scharr edge detection completed.")

    return abs_x, abs_y, magnitude
