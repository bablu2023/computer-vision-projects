import cv2


def draw_rectangle(
    image,
    top_left=(100, 100),
    bottom_right=(400, 400),
    color=(255, 0, 0),
    thickness=2
):
    """
    Draw a rectangle on an image.

    Parameters:
        image (numpy.ndarray): Input image.
        top_left (tuple): Top-left corner (x, y).
        bottom_right (tuple): Bottom-right corner (x, y).
        color (tuple): Rectangle color in BGR.
        thickness (int): Border thickness.
                         Use -1 for a filled rectangle.

    Returns:
        numpy.ndarray: Image with the rectangle drawn.
    """

    output = image.copy()

    cv2.rectangle(
        output,
        top_left,
        bottom_right,
        color,
        thickness
    )

    print(
        f"[INFO] Rectangle drawn from "
        f"{top_left} to {bottom_right}"
    )

    return output
