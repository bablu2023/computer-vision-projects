import cv2


def sift_feature_detection(image, nfeatures=500):
    """
    Detect SIFT keypoints and descriptors.

    Parameters:
        image (numpy.ndarray): Input BGR image.
        nfeatures (int): Maximum number of features.

    Returns:
        tuple:
            output_image,
            keypoints,
            descriptors
    """

    if not hasattr(cv2, "SIFT_create"):
        print("[WARNING] SIFT is not available in this OpenCV installation.")
        return image.copy(), [], None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sift = cv2.SIFT_create(
        nfeatures=nfeatures
    )

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    output = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print("[INFO] SIFT feature detection completed.")
    print(f"[INFO] Keypoints detected : {len(keypoints)}")

    if descriptors is not None:
        print(f"[INFO] Descriptor shape : {descriptors.shape}")
    else:
        print("[INFO] No descriptors found.")

    return output, keypoints, descriptors
