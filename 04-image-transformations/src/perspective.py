import cv2
import numpy as np


def perspective_transform(image):
    """
    Apply a perspective transformation to an image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Perspective transformed image.
    """

    height, width = image.shape[:2]

    # Four source points
    src_points = np.float32([
        [50, 50],
        [width - 50, 50],
        [50, height - 50],
        [width - 50, height - 50]
    ])

    # Four destination points
    dst_points = np.float32([
        [0, 0],
        [width, 50],
        [50, height],
        [width - 50, height - 50]
    ])

    # Compute perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(
        src_points,
        dst_points
    )

    transformed = cv2.warpPerspective(
        image,
        perspective_matrix,
        (width, height)
    )

    print("[INFO] Perspective transformation applied.")

    return transformed
