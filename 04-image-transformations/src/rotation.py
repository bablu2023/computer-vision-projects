import cv2


def rotate_image(image, angle=45, scale=1.0):
    """
    Rotate an image around its center.

    Parameters:
        image (numpy.ndarray): Input image.
        angle (float): Rotation angle in degrees.
                       Positive -> Counter-clockwise
                       Negative -> Clockwise
        scale (float): Scaling factor during rotation.

    Returns:
        numpy.ndarray: Rotated image.
    """

    height, width = image.shape[:2]

    center = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        scale
    )

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height)
    )

    print(
        f"[INFO] Image rotated "
        f"(angle={angle}°, scale={scale})"
    )

    return rotated
