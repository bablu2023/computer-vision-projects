import cv2


def rotate_image(image, angle, scale=1.0):
    """
    Rotate an image around its center.

    Parameters:
        image (numpy.ndarray): Input image.
        angle (float): Rotation angle in degrees.
                       Positive values rotate counterclockwise.
        scale (float): Scaling factor during rotation.

    Returns:
        numpy.ndarray: Rotated image.
    """

    # Get image dimensions
    (height, width) = image.shape[:2]

    # Compute the center of the image
    center = (width // 2, height // 2)

    # Create the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Apply the rotation
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height)
    )

    print(f"[INFO] Image rotated by {angle}° (scale={scale})")

    return rotated
