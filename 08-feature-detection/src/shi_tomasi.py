import cv2
import numpy as np


def shi_tomasi_detection(
    image,
    max_corners=200,
    quality_level=0.01,
    min_distance=10
):
    """
    Detect Shi-Tomasi corners.

    Parameters:
        image (numpy.ndarray): Input BGR image.
        max_corners (int): Maximum number of corners.
        quality_level (float): Minimum accepted quality.
        min_distance (int): Minimum distance between corners.

    Returns:
        numpy.ndarray: Image with detected corners.
    """

    output = image.copy()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )

    if corners is None:
        print("[INFO] No corners detected.")
        return output

    corners = np.int32(corners)

    for corner in corners:

        x, y = corner.ravel()

        cv2.circle(
            output,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

    print(f"[INFO] Shi-Tomasi corners detected: {len(corners)}")

    return output
