import cv2


def match_features(
    image1,
    keypoints1,
    descriptors1,
    image2,
    keypoints2,
    descriptors2,
    method="ORB",
    max_matches=50
):
    """
    Match feature descriptors between two images.

    Parameters:
        image1 (numpy.ndarray): First image.
        keypoints1 (list): Keypoints from first image.
        descriptors1 (numpy.ndarray): Descriptors from first image.
        image2 (numpy.ndarray): Second image.
        keypoints2 (list): Keypoints from second image.
        descriptors2 (numpy.ndarray): Descriptors from second image.
        method (str): 'ORB' or 'SIFT'.
        max_matches (int): Maximum matches to draw.

    Returns:
        numpy.ndarray: Image showing feature matches.
    """

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Descriptors cannot be None.")

    if method.upper() == "ORB":
        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=True
        )
    elif method.upper() == "SIFT":
        matcher = cv2.BFMatcher(
            cv2.NORM_L2,
            crossCheck=True
        )
    else:
        raise ValueError("Method must be 'ORB' or 'SIFT'.")

    matches = matcher.match(
        descriptors1,
        descriptors2
    )

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    matched_image = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches[:max_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print(f"[INFO] Total matches: {len(matches)}")
    print(f"[INFO] Displaying top {min(max_matches, len(matches))} matches.")

    return matched_image
