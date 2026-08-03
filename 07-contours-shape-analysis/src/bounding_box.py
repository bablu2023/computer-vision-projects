import cv2


def draw_bounding_boxes(
    image,
    contours,
    min_area=100,
    color=(0, 255, 0),
    thickness=2
):
    """
    Draw bounding boxes for contours larger than min_area.
    """

    output = image.copy()

    count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            thickness
        )

        print(
            f"[INFO] Bounding Box {count}: "
            f"Area={area:.2f}, "
            f"x={x}, y={y}, w={w}, h={h}"
        )

        count += 1

    print(f"[INFO] Total bounding boxes drawn: {count}")

    return output
