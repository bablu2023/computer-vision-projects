import cv2


def convert_to_gray(image):
    """
    Convert BGR image to Grayscale.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    print("[INFO] Converted to Grayscale.")

    return gray


def convert_to_rgb(image):
    """
    Convert BGR image to RGB.
    """

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    print("[INFO] Converted to RGB.")

    return rgb


def convert_to_hsv(image):
    """
    Convert BGR image to HSV.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    print("[INFO] Converted to HSV.")

    return hsv


def convert_to_lab(image):
    """
    Convert BGR image to LAB.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    print("[INFO] Converted to LAB.")

    return lab


def convert_to_ycrcb(image):
    """
    Convert BGR image to YCrCb.
    """

    ycrcb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2YCrCb
    )

    print("[INFO] Converted to YCrCb.")

    return ycrcb
