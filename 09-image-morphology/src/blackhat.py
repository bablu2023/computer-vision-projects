import cv2


def blackhat_image(
    image,
    kernel
):
    """
    Apply Black Hat transformation.

    Black Hat = Closing - Original Image

    Parameters:
        image (numpy.ndarray): Input image.
        kernel (numpy.ndarray): Structuring element.

    Returns:
        numpy.ndarray: Black Hat image.
    """

    blackhat = cv2.morphologyEx(
        image,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    print("[INFO] Black Hat transformation applied.")

    return blackhat
