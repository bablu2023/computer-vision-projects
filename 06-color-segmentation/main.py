from src.image_loader import load_image
from src.utils import print_image_info
from src.color_spaces import (
    convert_to_gray,
    convert_to_rgb,
    convert_to_hsv,
    convert_to_lab,
    convert_to_ycrcb,
)
from src.hsv_segmentation import segment_hsv
from src.masking import create_mask, apply_mask
from src.bitwise import bitwise_and, bitwise_not
from src.color_detection import detect_color
from src.save import save_image


def main():

    print("=" * 60)
    print("Project 6 : Color Spaces & Image Segmentation")
    print("=" * 60)

    # --------------------------------------------------
    # Load Image
    # --------------------------------------------------
    image = load_image("images/sample.jpg")
    print_image_info(image)

    # --------------------------------------------------
    # Color Space Conversions
    # --------------------------------------------------
    gray = convert_to_gray(image)
    rgb = convert_to_rgb(image)
    hsv = convert_to_hsv(image)
    lab = convert_to_lab(image)
    ycrcb = convert_to_ycrcb(image)

    save_image(gray, "gray.jpg")
    save_image(rgb, "rgb.jpg")
    save_image(hsv, "hsv.jpg")
    save_image(lab, "lab.jpg")
    save_image(ycrcb, "ycrcb.jpg")

    # --------------------------------------------------
    # HSV Segmentation
    # --------------------------------------------------
    lower_hsv = (0, 30, 60)
    upper_hsv = (25, 200, 255)

    _, hsv_mask, hsv_segment = segment_hsv(
        image,
        lower_hsv,
        upper_hsv
    )

    save_image(hsv_mask, "hsv_mask.jpg")
    save_image(hsv_segment, "hsv_segment.jpg")

    # --------------------------------------------------
    # BGR Masking
    # --------------------------------------------------
    lower_bgr = (0, 0, 100)
    upper_bgr = (120, 120, 255)

    mask = create_mask(
        image,
        lower_bgr,
        upper_bgr
    )

    masked = apply_mask(
        image,
        mask
    )

    save_image(mask, "mask.jpg")
    save_image(masked, "masked.jpg")

    # --------------------------------------------------
    # Bitwise Operations
    # --------------------------------------------------
    and_result = bitwise_and(
        image,
        mask
    )

    not_result = bitwise_not(image)

    save_image(and_result, "bitwise_and.jpg")
    save_image(not_result, "bitwise_not.jpg")

    # --------------------------------------------------
    # Color Detection
    # --------------------------------------------------
    detect_mask, detect_result = detect_color(
        image,
        lower_hsv,
        upper_hsv
    )

    save_image(detect_mask, "detected_mask.jpg")
    save_image(detect_result, "detected_color.jpg")

    print("\n" + "=" * 60)
    print("Project 6 Completed Successfully!")
    print("All output images saved in output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
