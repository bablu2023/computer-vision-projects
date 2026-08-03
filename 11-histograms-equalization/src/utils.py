def print_image_info(image):
    """
    Print basic image information.
    """

    height, width = image.shape[:2]

    channels = 1 if len(image.shape) == 2 else image.shape[2]

    print("\n" + "=" * 60)
    print("Image Information")
    print("=" * 60)
    print(f"Width      : {width}")
    print(f"Height     : {height}")
    print(f"Channels   : {channels}")
    print(f"Data Type  : {image.dtype}")
    print("=" * 60)
