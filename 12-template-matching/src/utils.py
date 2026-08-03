def print_image_info(image):
    """
    Print image information.
    """

    h, w = image.shape[:2]

    channels = 1 if len(image.shape) == 2 else image.shape[2]

    print("\n" + "=" * 60)
    print("Image Information")
    print("=" * 60)
    print(f"Width      : {w}")
    print(f"Height     : {h}")
    print(f"Channels   : {channels}")
    print(f"Data Type  : {image.dtype}")
    print("=" * 60)


def print_detection_summary(boxes):
    """
    Print detection summary.
    """

    print("\n" + "=" * 60)
    print("Detection Summary")
    print("=" * 60)

    print(f"Total Objects Detected : {len(boxes)}")

    for i, (x, y, w, h) in enumerate(boxes, start=1):
        print(
            f"{i:2d}: x={x}, y={y}, width={w}, height={h}"
        )

    print("=" * 60)
