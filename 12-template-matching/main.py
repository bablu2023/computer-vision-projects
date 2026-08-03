import os
import cv2

from src.image_loader import load_image
from src.template_loader import load_template
from src.template_matching import template_match
from src.localization import localize_template
from src.multi_detection import detect_multiple_objects
from src.nms import apply_nms
from src.save import save_image
from src.utils import (
    print_image_info,
    print_detection_summary
)


def create_template_if_missing():

    if os.path.exists("images/template.jpg"):
        return

    image = cv2.imread("images/sample.jpg")

    if image is None:
        raise FileNotFoundError("images/sample.jpg not found")

    template = image[
        150:250,
        150:250
    ]

    cv2.imwrite(
        "images/template.jpg",
        template
    )

    print("[INFO] Template automatically created.")


def main():

    print("=" * 70)
    print("Project 12 : Template Matching & Object Detection")
    print("=" * 70)

    os.makedirs("output", exist_ok=True)

    create_template_if_missing()

    image = load_image("images/sample.jpg")

    print_image_info(image)

    template = load_template("images/template.jpg")

    result, gray = template_match(
        image,
        template
    )

    # Single Detection
    localized, location, score = localize_template(
        image,
        template,
        result
    )

    save_image(
        localized,
        "single_detection.jpg"
    )

    # Multiple Detection
    multi_image, detections = detect_multiple_objects(
        image,
        template,
        result,
        threshold=0.80
    )

    save_image(
        multi_image,
        "multi_detection.jpg"
    )

    # Prepare scores for NMS
    scores = []

    for (x, y, w, h) in detections:
        scores.append(
            float(result[y, x])
        )

    nms_image, final_boxes = apply_nms(
        image,
        detections,
        scores,
        score_threshold=0.80,
        nms_threshold=0.30
    )

    save_image(
        nms_image,
        "nms.jpg"
    )

    print_detection_summary(final_boxes)

    print("\nBest Match")
    print("=" * 60)
    print(f"Location : {location}")
    print(f"Score    : {score:.4f}")

    print("\n" + "=" * 70)
    print("Project 12 Completed Successfully!")
    print("Results saved in output/")
    print("=" * 70)


if __name__ == "__main__":
    main()
