import cv2


def draw_convex_hulls(
    image,
    contours,
    min_area=100,
    color=(0, 255, 255),
    thickness=2
):
    """
    Draw convex hulls for detected contours.

    Parameters:
        image (numpy.ndarray): Input image.
        contours (list): List of contours.
        min_area (float): Ignore contours smaller than this area.
        color (tuple): Hull color (BGR).
        thickness (int): Line thickness.

    Returns:
        numpy.ndarray: Image with convex hulls.
    """

    output = image.copy()

    count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        hull = cv2.convexHull(contour)

        cv2.drawContours(
            output,
            [hull],
            -1,
            color,
            thickness
        )

        print(
            f"[INFO] Hull {count}: "
            f"Vertices={len(hull)}, "
            f"Area={area:.2f}"
        )

        count += 1

    print(f"[INFO] Total convex hulls drawn: {count}")

    return output
