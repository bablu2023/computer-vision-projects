import cv2
import numpy as np


def segment_hsv(image, lower_hsv, upper_hsv):
    """
    Segment an image based on an HSV color range.

    Parameters:
        image (numpy.ndarray): Input BGR image.
        lower_hsv (tuple/list): Lower HSV bound.
        upper_hsv (tuple/list): Upper HSV bound.

    Returns:
        tuple:
            hsv_image - HSV converted image
            mask - Binary mask
            segmented - Color segmented output
    """

    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)

    mask = cv2.inRange(
        hsv_image,
        lower,
        upper
    )

    segmented = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    print("[INFO] HSV segmentation completed.")

    return hsv_image, mask, segmented
