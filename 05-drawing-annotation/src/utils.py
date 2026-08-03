import cv2


def get_image_info(image):
    """
    Get basic information about an image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        dict: Image information.
    """

    height, width = image.shape[:2]

    channels = 1 if len(image.shape) == 2 else image.shape[2]

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": image.dtype,
        "size": image.size
    }


def print_image_info(image):
    """
    Print image information.
    """

    info = get_image_info(image)

    print("\n========== Image Information ==========")
    print(f"Width       : {info['width']}")
    print(f"Height      : {info['height']}")
    print(f"Channels    : {info['channels']}")
    print(f"Data Type   : {info['dtype']}")
    print(f"Total Pixels: {info['size']}")
    print("=======================================\n")


def get_aspect_ratio(image):
    """
    Return image aspect ratio.
    """

    h, w = image.shape[:2]

    return w / h


def is_grayscale(image):
    """
    Check whether an image is grayscale.
    """

    return len(image.shape) == 2


def copy_image(image):
    """
    Return a deep copy of the image.
    """

    return image.copy()


def convert_to_rgb(image):
    """
    Convert BGR image to RGB.
    """

    if is_grayscale(image):
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_gray(image):
    """
    Convert BGR image to Grayscale.
    """

    if is_grayscale(image):
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
