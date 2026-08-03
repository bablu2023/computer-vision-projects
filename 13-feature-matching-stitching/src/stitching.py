import cv2
import numpy as np


def stitch_images(image1, image2, H):
    """
    Stitch two images using a homography matrix.

    Parameters:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.
        H (numpy.ndarray): Homography matrix.

    Returns:
        numpy.ndarray: Panorama image.
    """

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    panorama = cv2.warpPerspective(
        image1,
        H,
        (w1 + w2, max(h1, h2))
    )

    panorama[0:h2, 0:w2] = image2

    print("[INFO] Image stitching completed.")
    print(f"[INFO] Panorama size : {panorama.shape[1]} x {panorama.shape[0]}")

    return panorama
