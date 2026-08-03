import cv2


def apply_nms(
    image,
    detections,
    scores=None,
    score_threshold=0.5,
    nms_threshold=0.4
):
    """
    Apply Non-Maximum Suppression (NMS).

    Parameters:
        image (numpy.ndarray): Original image.
        detections (list): List of (x, y, w, h).
        scores (list): Confidence scores.
        score_threshold (float): Minimum confidence.
        nms_threshold (float): IoU threshold.

    Returns:
        tuple:
            output_image
            final_boxes
    """

    output = image.copy()

    if len(detections) == 0:
        print("[INFO] No detections found.")
        return output, []

    if scores is None:
        scores = [1.0] * len(detections)

    indices = cv2.dnn.NMSBoxes(
        detections,
        scores,
        score_threshold,
        nms_threshold
    )

    final_boxes = []

    if len(indices) > 0:

        for idx in indices.flatten():

            x, y, w, h = detections[idx]

            final_boxes.append(
                (x, y, w, h)
            )

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

    print(f"[INFO] Original detections : {len(detections)}")
    print(f"[INFO] After NMS           : {len(final_boxes)}")

    return output, final_boxes
