def crop_image(image, x, y, width, height):
    """
    Crop a rectangular region from an image.

    Parameters:
        image (numpy.ndarray): Input image.
        x (int): Top-left x-coordinate.
        y (int): Top-left y-coordinate.
        width (int): Width of the crop.
        height (int): Height of the crop.

    Returns:
        numpy.ndarray: Cropped image.

    Raises:
        ValueError: If the crop region is outside the image.
    """

    img_height, img_width = image.shape[:2]

    # Validate crop boundaries
    if x < 0 or y < 0:
        raise ValueError("x and y must be non-negative.")

    if x + width > img_width or y + height > img_height:
        raise ValueError("Crop region exceeds image boundaries.")

    cropped = image[y:y + height, x:x + width]

    print(
        f"[INFO] Image cropped: "
        f"x={x}, y={y}, width={width}, height={height}"
    )

    return cropped
