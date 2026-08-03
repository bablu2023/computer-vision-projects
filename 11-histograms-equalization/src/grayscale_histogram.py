import cv2
import numpy as np


def grayscale_histogram(gray):
    """
    Compute the grayscale histogram.

    Parameters:
        gray (numpy.ndarray): Grayscale image.

    Returns:
        numpy.ndarray: Histogram (256x1).
    """

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    print("[INFO] Grayscale histogram computed.")
    print(f"[INFO] Histogram shape: {histogram.shape}")
    print(f"[INFO] Total pixels: {int(np.sum(histogram))}")

    return histogram
