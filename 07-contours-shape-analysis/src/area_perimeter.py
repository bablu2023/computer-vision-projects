import cv2


def contour_properties(contours):
    """
    Compute the area and perimeter of each contour.

    Parameters:
        contours (list): List of detected contours.

    Returns:
        list: List of dictionaries containing contour properties.
    """

    properties = []

    for index, contour in enumerate(contours):

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(
            contour,
            True
        )

        properties.append({
            "id": index,
            "area": area,
            "perimeter": perimeter
        })

        print(
            f"[INFO] Contour {index}: "
            f"Area={area:.2f}, "
            f"Perimeter={perimeter:.2f}"
        )

    return properties
