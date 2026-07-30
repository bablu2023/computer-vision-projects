def print_image_info(image):
    """
    Print basic information about an image.

    Parameters:
        image (numpy.ndarray): Input image.
    """
    height, width = image.shape[:2]

    if len(image.shape) == 3:
        channels = image.shape[2]
    else:
        channels = 1

    print("\n========== Image Information ==========")
    print(f"Width      : {width} pixels")
    print(f"Height     : {height} pixels")
    print(f"Channels   : {channels}")
    print(f"Shape      : {image.shape}")
    print(f"Data Type  : {image.dtype}")
    print(f"Total Size : {image.size} values")
    print("=======================================\n")
