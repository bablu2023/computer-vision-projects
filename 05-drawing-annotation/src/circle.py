import cv2


def draw_circle(
    image,
    center=(256, 256),
    radius=100,
    color=(0, 0, 255),
    thickness=2
):
    """
    Draw a circle on an image.

    Parameters:
        image (numpy.ndarray): Input image.
        center (tuple): Center coordinates (x, y).
        radius (int): Circle radius in pixels.
        color (tuple): Circle color in BGR format.
        thickness (int): Border thickness.
                         Use -1 for a filled circle.

    Returns:
        numpy.ndarray: Image with the circle drawn.
    """

    output = image.copy()

    cv2.circle(
        output,
        center,
        radius,
        color,
        thickness
    )

    print(
        f"[INFO] Circle drawn at {center} "
        f"(radius={radius})"
    )

    return output
