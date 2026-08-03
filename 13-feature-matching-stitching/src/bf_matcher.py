import cv2


def bf_match_features(
    image1,
    keypoints1,
    descriptors1,
    image2,
    keypoints2,
    descriptors2,
    norm_type=cv2.NORM_HAMMING,
    cross_check=True
):
    """
    Match feature descriptors using Brute Force Matcher.

    Parameters:
        image1 (numpy.ndarray): First image.
        keypoints1 (list): Keypoints from first image.
        descriptors1 (numpy.ndarray): Descriptors from first image.
        image2 (numpy.ndarray): Second image.
        keypoints2 (list): Keypoints from second image.
        descriptors2 (numpy.ndarray): Descriptors from second image.
        norm_type (int): Distance metric.
        cross_check (bool): Enable cross-checking.

    Returns:
        tuple:
            matched_image,
            matches
    """

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Descriptors cannot be None.")

    matcher = cv2.BFMatcher(
        normType=norm_type,
        crossCheck=cross_check
    )

    matches = matcher.match(
        descriptors1,
        descriptors2
    )

    matches = sorted(
        matches,
        key=lambda match: match.distance
    )

    matched_image = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print("[INFO] BF Matcher completed.")
    print(f"[INFO] Total matches : {len(matches)}")
    print("[INFO] Showing top 50 matches.")

    return matched_image, matches
