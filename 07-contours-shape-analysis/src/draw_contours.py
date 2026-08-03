import cv2


def draw_all_contours(
    image,
    contours,
    color=(0, 255, 0),
    thickness=2
):
    """
    Draw all detected contours on an image.

    Parameters:
        image (numpy.ndarray): Input image.
        contours (list): List of contours.
        color (tuple): Contour color (BGR).
        thickness (int): Line thickness.

    Returns:
        numpy.ndarray: Annotated image.
    """

    output = image.copy()

    cv2.drawContours(
        output,
        contours,
        -1,
        color,
        thickness
    )

    print(f"[INFO] Drew {len(contours)} contour(s).")

    return output
