import cv2


def scale_image(image, scale_x=1.5, scale_y=1.5, interpolation=cv2.INTER_LINEAR):
    """
    Scale an image.

    Parameters:
        image (numpy.ndarray): Input image.
        scale_x (float): Horizontal scaling factor.
        scale_y (float): Vertical scaling factor.
        interpolation (int): OpenCV interpolation method.

    Returns:
        numpy.ndarray: Scaled image.
    """

    scaled = cv2.resize(
        image,
        None,
        fx=scale_x,
        fy=scale_y,
        interpolation=interpolation
    )

    print(
        f"[INFO] Image scaled "
        f"(scale_x={scale_x}, scale_y={scale_y})"
    )

    return scaled
