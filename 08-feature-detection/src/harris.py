import cv2
import numpy as np


def harris_corner_detection(
    image,
    block_size=2,
    ksize=3,
    k=0.04,
    threshold_ratio=0.01
):
    """
    Detect Harris corners and mark them on the image.

    Parameters:
        image (numpy.ndarray): Input BGR image.
        block_size (int): Neighborhood size.
        ksize (int): Sobel kernel size.
        k (float): Harris detector parameter.
        threshold_ratio (float): Corner threshold ratio.

    Returns:
        numpy.ndarray: Image with detected corners.
    """

    output = image.copy()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.float32(gray)

    response = cv2.cornerHarris(
        gray,
        block_size,
        ksize,
        k
    )

    response = cv2.dilate(
        response,
        None
    )

    threshold = threshold_ratio * response.max()

    output[response > threshold] = [0, 0, 255]

    corners = np.sum(response > threshold)

    print(f"[INFO] Harris corners detected: {corners}")

    return output
