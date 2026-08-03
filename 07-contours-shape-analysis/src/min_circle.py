import cv2


def draw_min_enclosing_circles(
    image,
    contours,
    min_area=100,
    color=(255, 0, 0),
    thickness=2
):
    """
    Draw the minimum enclosing circle for each contour.

    Parameters:
        image (numpy.ndarray): Input image.
        contours (list): List of contours.
        min_area (float): Ignore contours smaller than this area.
        color (tuple): Circle color (BGR).
        thickness (int): Circle thickness.

    Returns:
        numpy.ndarray: Image with enclosing circles.
    """

    output = image.copy()

    count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)

        center = (int(x), int(y))
        radius = int(radius)

        cv2.circle(
            output,
            center,
            radius,
            color,
            thickness
        )

        print(
            f"[INFO] Circle {count}: "
            f"Center={center}, Radius={radius}, Area={area:.2f}"
        )

        count += 1

    print(f"[INFO] Total enclosing circles drawn: {count}")

    return output
