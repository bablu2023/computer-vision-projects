def print_image_info(image, name="Image"):
    """
    Print image information.
    """

    h, w = image.shape[:2]

    channels = (
        1 if len(image.shape) == 2
        else image.shape[2]
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"Width      : {w}")
    print(f"Height     : {h}")
    print(f"Channels   : {channels}")
    print(f"Data Type  : {image.dtype}")
    print("=" * 60)


def print_match_statistics(matches):
    """
    Print feature matching statistics.
    """

    print("\n" + "=" * 60)
    print("Feature Matching Statistics")
    print("=" * 60)

    print(f"Total Matches : {len(matches)}")

    if len(matches) > 0:
        print(f"Best Distance : {matches[0].distance:.2f}")
        print(f"Worst Distance: {matches[-1].distance:.2f}")

    print("=" * 60)
