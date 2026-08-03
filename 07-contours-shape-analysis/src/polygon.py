import cv2


def draw_polygons(
    image,
    contours,
    min_area=100,
    epsilon_factor=0.02,
    color=(0, 0, 255),
    thickness=2
):
    """
    Approximate contours as polygons and draw them.

    Parameters:
        image (numpy.ndarray): Input image.
        contours (list): List of contours.
        min_area (float): Ignore contours smaller than this area.
        epsilon_factor (float): Approximation accuracy factor.
        color (tuple): Polygon color (BGR).
        thickness (int): Line thickness.

    Returns:
        numpy.ndarray: Image with polygon approximations.
    """

    output = image.copy()

    count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        epsilon = epsilon_factor * perimeter

        polygon = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        cv2.polylines(
            output,
            [polygon],
            True,
            color,
            thickness
        )

        print(
            f"[INFO] Polygon {count}: "
            f"Vertices={len(polygon)}, "
            f"Area={area:.2f}"
        )

        count += 1

    print(f"[INFO] Total polygons drawn: {count}")

    return output
