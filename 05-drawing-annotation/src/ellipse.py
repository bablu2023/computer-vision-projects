import cv2


def draw_ellipse(
    image,
    center=(256, 256),
    axes=(120, 60),
    angle=30,
    start_angle=0,
    end_angle=360,
    color=(255, 0, 255),
    thickness=2
):
    """
    Draw an ellipse on an image.

    Parameters:
        image (numpy.ndarray): Input image.
        center (tuple): Center coordinates (x, y).
        axes (tuple): Lengths of major and minor axes.
        angle (float): Rotation angle of the ellipse.
        start_angle (float): Starting angle of the arc.
        end_angle (float): Ending angle of the arc.
        color (tuple): Ellipse color in BGR format.
        thickness (int): Border thickness.
                         Use -1 for a filled ellipse.

    Returns:
        numpy.ndarray: Image with the ellipse drawn.
    """

    output = image.copy()

    cv2.ellipse(
        output,
        center,
        axes,
        angle,
        start_angle,
        end_angle,
        color,
        thickness
    )

    print(
        f"[INFO] Ellipse drawn at {center} "
        f"(axes={axes}, angle={angle}°)"
    )

    return output
