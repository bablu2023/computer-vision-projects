import cv2
import numpy as np


def sobel_edge_detection(
    gray,
    kernel_size=3
):
    """
    Apply Sobel edge detection.

    Parameters:
        gray (numpy.ndarray): Grayscale input image.
        kernel_size (int): Sobel kernel size (1, 3, 5, or 7).

    Returns:
        tuple:
            sobel_x
            sobel_y
            sobel_magnitude
    """

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=kernel_size
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=kernel_size
    )

    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)

    magnitude = cv2.magnitude(
        sobel_x,
        sobel_y
    )

    magnitude = cv2.convertScaleAbs(
        magnitude
    )

    print("[INFO] Sobel edge detection completed.")
    print(f"[INFO] Kernel size: {kernel_size}")

    return abs_x, abs_y, magnitude
