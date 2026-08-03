import cv2


def get_image_info(image):
    """
    Return image information as a dictionary.
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
    print(f"Width      : {info['width']}")
    print(f"Height     : {info['height']}")
    print(f"Channels   : {info['channels']}")
    print(f"Data Type  : {info['dtype']}")
    print(f"Total Size : {info['size']}")
    print("=======================================\n")


def convert_to_gray(image):
    """
    Convert BGR image to grayscale.
    """

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def convert_to_rgb(image):
    """
    Convert BGR image to RGB.
    """

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
