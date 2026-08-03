import cv2


def gradient_image(
    image,
    kernel
):
    """
    Apply morphological gradient.

    Gradient = Dilation - Erosion

    Parameters:
        image (numpy.ndarray): Binary input image.
        kernel (numpy.ndarray): Structuring element.

    Returns:
        numpy.ndarray: Gradient image.
    """

    gradient = cv2.morphologyEx(
        image,
        cv2.MORPH_GRADIENT,
        kernel
    )

    print("[INFO] Morphological gradient applied.")

    return gradient
