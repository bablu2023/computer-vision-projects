import cv2


def preprocess_image(image):
    """
    Convert a BGR image to grayscale.

    Parameters:
        image (numpy.ndarray): Input BGR image.

    Returns:
        numpy.ndarray: Grayscale image.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    print("[INFO] Converted image to grayscale.")

    return gray
