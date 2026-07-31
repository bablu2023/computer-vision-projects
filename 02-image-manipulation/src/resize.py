import cv2


def resize_image(image, width=None, height=None, scale=None,
                 interpolation=cv2.INTER_LINEAR):
    """
    Resize an image.

    Parameters:
        image (numpy.ndarray): Input image.
        width (int, optional): Desired width in pixels.
        height (int, optional): Desired height in pixels.
        scale (float, optional): Scaling factor (e.g., 0.5 or 2.0).
        interpolation (int): OpenCV interpolation method.

    Returns:
        numpy.ndarray: Resized image.

    Raises:
        ValueError: If neither scale nor width/height is provided.
    """

    # Resize using scaling factor
    if scale is not None:
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation
        )

        print(f"[INFO] Image resized using scale factor {scale}")
        return resized

    # Resize using explicit dimensions
    if width is not None and height is not None:
        resized = cv2.resize(
            image,
            (width, height),
            interpolation=interpolation
        )

        print(f"[INFO] Image resized to {width} x {height}")
        return resized

    raise ValueError(
        "Provide either scale or both width and height."
    )
