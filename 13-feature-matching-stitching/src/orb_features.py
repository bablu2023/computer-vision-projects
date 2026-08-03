import cv2


def orb_feature_detection(image, nfeatures=500):
    """
    Detect ORB keypoints and descriptors.

    Parameters:
        image (numpy.ndarray): Input BGR image.
        nfeatures (int): Maximum number of features.

    Returns:
        tuple:
            output_image,
            keypoints,
            descriptors
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=nfeatures
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

    print("[INFO] ORB feature detection completed.")
    print(f"[INFO] Keypoints detected : {len(keypoints)}")

    if descriptors is not None:
        print(f"[INFO] Descriptor shape : {descriptors.shape}")
    else:
        print("[INFO] No descriptors found.")

    return output, keypoints, descriptors
