import cv2


def brisk_feature_detection(image, threshold=30, octaves=3):
    """
    BRISK Feature Detection.

    Returns:
        None if BRISK is unavailable.
    """

    if not hasattr(cv2, "BRISK_create"):
        print("[WARNING] BRISK is not available in this OpenCV build.")
        return None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brisk = cv2.BRISK_create(
        thresh=threshold,
        octaves=octaves
    )

    keypoints, descriptors = brisk.detectAndCompute(
        gray,
        None
    )

    output = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(255, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"[INFO] BRISK keypoints detected: {len(keypoints)}")

    return output, keypoints, descriptors
