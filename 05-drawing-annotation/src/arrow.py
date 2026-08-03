import cv2


def draw_arrow(
    image,
    start_point=(100, 100),
    end_point=(400, 300),
    color=(0, 255, 0),
    thickness=2,
    tip_length=0.2
):
    """
    Draw an arrow on an image.

    Parameters:
        image (numpy.ndarray): Input image.
        start_point (tuple): Arrow starting point (x, y).
        end_point (tuple): Arrow ending point (x, y).
        color (tuple): Arrow color in BGR format.
        thickness (int): Arrow thickness.
        tip_length (float): Length of the arrow tip (0.0–1.0).

    Returns:
        numpy.ndarray: Image with the arrow drawn.
    """

    output = image.copy()

    cv2.arrowedLine(
        output,
        start_point,
        end_point,
        color,
        thickness,
        cv2.LINE_AA,
        0,
        tip_length
    )

    print(
        f"[INFO] Arrow drawn from "
        f"{start_point} to {end_point}"
    )

    return output
