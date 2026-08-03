import cv2


def draw_centroids(
    image,
    contours,
    min_area=100,
    color=(0, 0, 255),
    radius=5
):
    """
    Compute and draw contour centroids.

    Parameters:
        image (numpy.ndarray): Input image.
        contours (list): List of contours.
        min_area (float): Ignore contours smaller than this area.
        color (tuple): Centroid color (BGR).
        radius (int): Circle radius.

    Returns:
        numpy.ndarray: Image with centroids.
    """

    output = image.copy()

    count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        M = cv2.moments(contour)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(
            output,
            (cx, cy),
            radius,
            color,
            -1
        )

        cv2.putText(
            output,
            f"C{count}",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

        print(
            f"[INFO] Contour {count}: "
            f"Centroid=({cx}, {cy}), "
            f"Area={area:.2f}"
        )

        count += 1

    print(f"[INFO] Total centroids drawn: {count}")

    return output
