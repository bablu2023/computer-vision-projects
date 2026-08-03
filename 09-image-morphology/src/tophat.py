import cv2


def tophat_image(
    image,
    kernel
):
    """
    Apply Top Hat transformation.

    Top Hat = Original Image - Opening

    Parameters:
        image (numpy.ndarray): Input image.
        kernel (numpy.ndarray): Structuring element.

    Returns:
        numpy.ndarray: Top Hat image.
    """

    tophat = cv2.morphologyEx(
        image,
        cv2.MORPH_TOPHAT,
        kernel
    )

    print("[INFO] Top Hat transformation applied.")

    return tophat
