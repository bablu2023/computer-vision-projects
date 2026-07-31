import cv2
import numpy as np


def affine_transform(image):
    """
    Apply an affine transformation to an image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Affine transformed image.
    """

    height, width = image.shape[:2]

    # Three source points
    src_points = np.float32([
        [50, 50],
        [200, 50],
        [50, 200]
    ])

    # Corresponding destination points
    dst_points = np.float32([
        [10, 100],
        [200, 50],
        [100, 250]
    ])

    # Compute affine transformation matrix
    affine_matrix = cv2.getAffineTransform(
        src_points,
        dst_points
    )

    transformed = cv2.warpAffine(
        image,
        affine_matrix,
        (width, height)
    )

    print("[INFO] Affine transformation applied.")

    return transformed
