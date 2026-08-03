import cv2


def akaze_feature_detection(image):
    """
    Detect AKAZE keypoints and descriptors.

    Returns:
        tuple:
            output
            keypoints
            descriptors
    """

    if not hasattr(cv2, "AKAZE_create"):
        print("[WARNING] AKAZE is not available in this OpenCV build.")
        return None, None, None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    akaze = cv2.AKAZE_create()

    keypoints, descriptors = akaze.detectAndCompute(
        gray,
        None
    )

    output = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(255, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"[INFO] AKAZE keypoints detected: {len(keypoints)}")

    if descriptors is not None:
        print(f"[INFO] Descriptor shape: {descriptors.shape}")

    return output, keypoints, descriptors
