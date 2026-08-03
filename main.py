from src.image_loader import load_image
from src.sift_features import sift_feature_detection
from src.ratio_test import lowe_ratio_test

import cv2
import os

os.makedirs("output", exist_ok=True)

image1 = load_image("images/sample.jpg")
image2 = load_image("images/sample2.jpg")

_, kp1, des1 = sift_feature_detection(image1)
_, kp2, des2 = sift_feature_detection(image2)

matcher = cv2.BFMatcher(cv2.NORM_L2)

raw_matches = matcher.knnMatch(
    des1,
    des2,
    k=2
)

good_matches = lowe_ratio_test(
    raw_matches,
    ratio=0.75
)

matched = cv2.drawMatches(
    image1,
    kp1,
    image2,
    kp2,
    good_matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    "output/ratio_test.jpg",
    matched
)

print("\nRatio Test Statistics")
print("=" * 50)
print(f"Raw Matches  : {len(raw_matches)}")
print(f"Good Matches : {len(good_matches)}")
print("=" * 50)
