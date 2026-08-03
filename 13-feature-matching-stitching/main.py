import os
import cv2

from src.image_loader import load_image
from src.sift_features import sift_feature_detection
from src.ratio_test import lowe_ratio_test
from src.homography import estimate_homography
from src.stitching import stitch_images
from src.save import save_image
from src.utils import (
    print_image_info,
    print_match_statistics
)

print("=" * 70)
print("Project 13 : Feature Matching & Image Stitching")
print("=" * 70)

os.makedirs("output", exist_ok=True)

image1 = load_image("images/sample.jpg")
image2 = load_image("images/sample2.jpg")

print_image_info(image1, "Image 1")
print_image_info(image2, "Image 2")

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

H, mask = estimate_homography(
    kp1,
    kp2,
    good_matches
)

panorama = stitch_images(
    image1,
    image2,
    H
)

matched_image = cv2.drawMatches(
    image1,
    kp1,
    image2,
    kp2,
    good_matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

save_image(
    matched_image,
    "feature_matches.jpg"
)

save_image(
    panorama,
    "panorama.jpg"
)

print_match_statistics(good_matches)

print("\nHomography Matrix")
print("=" * 60)
print(H)

print("\n" + "=" * 70)
print("Project 13 Completed Successfully!")
print("Results saved in output/")
print("=" * 70)
