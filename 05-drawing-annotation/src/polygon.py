import cv2
import numpy as np


def draw_polygon(
    image,
    points=None,
    color=(0, 255, 255),
    thickness=2,
    filled=False
):
    """
    Draw a polygon on an image.

    Parameters:
        image (numpy.ndarray): Input image.
        points (list): List of (x, y) polygon vertices.
        color (tuple): Polygon color in BGR format.
        thickness (int): Border thickness.
        filled (bool): Fill the polygon if True.

    Returns:
        numpy.ndarray: Image with the polygon drawn.
    """

    if points is None:
        points = [
            (120, 100),
            (400, 120),
            (450, 300),
            (300, 420),
            (120, 350)
        ]

    output = image.copy()

    pts = np.array(points, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))

    if filled:
        cv2.fillPoly(
            output,
            [pts],
            color
        )
    else:
        cv2.polylines(
            output,
            [pts],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )

    print(f"[INFO] Polygon drawn ({len(points)} vertices)")

    return output
