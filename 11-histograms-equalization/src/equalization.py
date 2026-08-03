import cv2


def histogram_equalization(gray):
    """
    Apply histogram equalization.

    Parameters:
        gray (numpy.ndarray): Grayscale image.

    Returns:
        numpy.ndarray: Equalized image.
    """

    equalized = cv2.equalizeHist(
        gray
    )

    print("[INFO] Histogram equalization completed.")

    return equalized
