import cv2


def get_image_info(image):
    """
    Return image information.
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


def filter_contours(
    contours,
    min_area=100
):
    """
    Filter contours by minimum area.

    Parameters:
        contours (list): Input contours.
        min_area (float): Minimum contour area.

    Returns:
        list: Filtered contours.
    """

    filtered = []

    for contour in contours:

        if cv2.contourArea(contour) >= min_area:
            filtered.append(contour)

    print(
        f"[INFO] Filtered contours: "
        f"{len(filtered)} / {len(contours)}"
    )

    return filtered
