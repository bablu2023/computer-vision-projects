import cv2


def detect_shapes(
    image,
    contours,
    min_area=100
):
    """
    Detect and classify geometric shapes.

    Parameters:
        image (numpy.ndarray): Input image.
        contours (list): List of contours.
        min_area (float): Ignore contours smaller than this area.

    Returns:
        numpy.ndarray: Image annotated with detected shapes.
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

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        vertices = len(approx)

        x, y, w, h = cv2.boundingRect(approx)

        aspect_ratio = w / float(h)

        if vertices == 3:
            shape = "Triangle"

        elif vertices == 4:

            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"

        elif vertices == 5:
            shape = "Pentagon"

        elif vertices == 6:
            shape = "Hexagon"

        elif vertices > 6:
            shape = "Circle"

        else:
            shape = "Unknown"

        cv2.drawContours(
            output,
            [approx],
            -1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            shape,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

        print(
            f"[INFO] Shape {count}: "
            f"{shape} "
            f"(Vertices={vertices}, Area={area:.2f})"
        )

        count += 1

    print(f"[INFO] Total detected shapes: {count}")

    return output
