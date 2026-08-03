import cv2
import numpy as np


def estimate_homography(
    keypoints1,
    keypoints2,
    matches,
    reprojection_threshold=5.0
):
    """
    Estimate homography using matched keypoints.

    Parameters:
        keypoints1 (list): Keypoints from image 1.
        keypoints2 (list): Keypoints from image 2.
        matches (list): Good matches.
        reprojection_threshold (float): RANSAC threshold.

    Returns:
        tuple:
            H
            mask
    """

    if len(matches) < 4:
        raise ValueError(
            "At least 4 matches are required."
        )

    src_pts = np.float32([
        keypoints1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        keypoints2[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        reprojection_threshold
    )

    inliers = int(mask.sum()) if mask is not None else 0

    print("[INFO] Homography estimated.")
    print(f"[INFO] Total matches : {len(matches)}")
    print(f"[INFO] Inliers       : {inliers}")

    return H, mask
