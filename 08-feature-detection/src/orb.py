import cv2


def orb_feature_detection(
    image,
    max_features=500
):
    """
    Detect ORB keypoints and descriptors.

    Parameters:
        image (numpy.ndarray): Input BGR image.
        max_features (int): Maximum number of features.

    Returns:
        tuple:
            output      - Image with keypoints
            keypoints   - Detected keypoints
            descriptors - ORB descriptors
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=max_features
    )

    keypoints, descriptors = orb.detectAndCompute(
        gray,
        None
    )

    output = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(f"[INFO] ORB keypoints detected: {len(keypoints)}")

    if descriptors is not None:
        print(f"[INFO] Descriptor shape: {descriptors.shape}")
    else:
        print("[INFO] No descriptors generated.")

    return output, keypoints, descriptors
