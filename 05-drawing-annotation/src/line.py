import cv2


def draw_line(
    image,
    start_point=(50, 50),
    end_point=(450, 450),
    color=(0, 255, 0),
    thickness=2,
    line_type=cv2.LINE_AA
):
    """
    Draw a line on an image.

    Parameters:
        image (numpy.ndarray): Input image.
        start_point (tuple): Starting coordinate (x, y).
        end_point (tuple): Ending coordinate (x, y).
        color (tuple): Line color in BGR format.
        thickness (int): Line thickness.
        line_type (int): OpenCV line type.

    Returns:
        numpy.ndarray: Image with the line drawn.
    """

    output = image.copy()

    cv2.line(
        output,
        start_point,
        end_point,
        color,
        thickness,
        line_type
    )

    print(
        f"[INFO] Line drawn from {start_point} to {end_point}"
    )

    return output
