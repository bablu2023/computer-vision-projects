import cv2


def fast_feature_detection(
    image,
    threshold=25,
    nonmax_suppression=True
):
    """
    Detect FAST keypoints.

    Parameters:
        image (numpy.ndarray): Input BGR image.
        threshold (int): FAST intensity threshold.
        nonmax_suppression (bool): Enable non-max suppression.

    Returns:
        tuple:
            output - Image with FAST keypoints
            keypoints - Detected keypoints
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    detector = cv2.FastFeatureDetector_create(
        threshold=threshold,
        nonmaxSuppression=nonmax_suppression
    )

    keypoints = detector.detect(
        gray,
        None
    )

    output = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"[INFO] FAST keypoints detected: {len(keypoints)}")
    print(f"[INFO] Threshold: {threshold}")
    print(f"[INFO] Non-max suppression: {nonmax_suppression}")

    return output, keypoints
