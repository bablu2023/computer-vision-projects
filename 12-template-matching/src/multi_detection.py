import cv2
import numpy as np


def detect_multiple_objects(
    image,
    template,
    result,
    threshold=0.8
):
    """
    Detect multiple template matches.

    Parameters:
        image (numpy.ndarray): Original BGR image.
        template (numpy.ndarray): Template image.
        result (numpy.ndarray): Response from matchTemplate().
        threshold (float): Matching threshold.

    Returns:
        tuple:
            output_image
            detections
    """

    output = image.copy()

    h, w = template.shape[:2]

    locations = np.where(result >= threshold)

    detections = []

    for pt in zip(*locations[::-1]):

        detections.append(
            (
                pt[0],
                pt[1],
                w,
                h
            )
        )

        cv2.rectangle(
            output,
            pt,
            (
                pt[0] + w,
                pt[1] + h
            ),
            (0, 255, 0),
            2
        )

    print(f"[INFO] Detection threshold : {threshold}")
    print(f"[INFO] Total detections    : {len(detections)}")

    return output, detections
