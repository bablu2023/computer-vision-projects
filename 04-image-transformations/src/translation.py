import cv2
import numpy as np


def translate_image(image, tx=50, ty=50):
    """
    Translate (shift) an image.

    Parameters:
        image (numpy.ndarray): Input image.
        tx (int): Horizontal shift in pixels.
                  Positive -> Right
                  Negative -> Left
        ty (int): Vertical shift in pixels.
                  Positive -> Down
                  Negative -> Up

    Returns:
        numpy.ndarray: Translated image.
    """

    height, width = image.shape[:2]

    # Translation matrix
    translation_matrix = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    translated = cv2.warpAffine(
        image,
        translation_matrix,
        (width, height)
    )

    print(f"[INFO] Image translated (tx={tx}, ty={ty})")

    return translated
