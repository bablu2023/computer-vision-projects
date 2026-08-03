import cv2
import numpy as np


def flann_match_features(
    image1,
    keypoints1,
    descriptors1,
    image2,
    keypoints2,
    descriptors2,
    ratio=0.75
):
    """
    Match SIFT descriptors using FLANN.

    Parameters:
        image1 (numpy.ndarray): First image.
        keypoints1 (list): Keypoints from first image.
        descriptors1 (numpy.ndarray): Descriptors from first image.
        image2 (numpy.ndarray): Second image.
        keypoints2 (list): Keypoints from second image.
        descriptors2 (numpy.ndarray): Descriptors from second image.
        ratio (float): Lowe's ratio threshold.

    Returns:
        tuple:
            matched_image,
            good_matches
    """

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Descriptors cannot be None.")

    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    index_params = dict(
        algorithm=1,   # FLANN_INDEX_KDTREE
        trees=5
    )

    search_params = dict(
        checks=50
    )

    matcher = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    knn_matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good_matches.append(m)

    matched_image = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        good_matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print("[INFO] FLANN matching completed.")
    print(f"[INFO] Raw matches : {len(knn_matches)}")
    print(f"[INFO] Good matches: {len(good_matches)}")
    print(f"[INFO] Ratio test : {ratio}")

    return matched_image, good_matches
