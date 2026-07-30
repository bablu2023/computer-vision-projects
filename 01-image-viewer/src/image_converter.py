import cv2


def convert_to_grayscale(image):
    """
    Convert a BGR image to grayscale.

    Parameters:
        image (numpy.ndarray): Input color image.

    Returns:
        numpy.ndarray: Grayscale image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("[INFO] Image converted to grayscale.")

    return gray_image
