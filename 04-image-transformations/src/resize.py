import cv2


def resize_image(image, width=None, height=None, interpolation=cv2.INTER_LINEAR):
    """
    Resize an image while preserving the aspect ratio.

    Parameters:
        image (numpy.ndarray): Input image.
        width (int): Desired width.
        height (int): Desired height.
        interpolation (int): OpenCV interpolation method.

    Returns:
        numpy.ndarray: Resized image.
    """

    if width is None and height is None:
        return image

    h, w = image.shape[:2]

    if width is None:
        ratio = height / float(h)
        width = int(w * ratio)

    elif height is None:
        ratio = width / float(w)
        height = int(h * ratio)

    resized = cv2.resize(
        image,
        (width, height),
        interpolation=interpolation
    )

    print(f"[INFO] Image resized to {width} x {height}")

    return resized
