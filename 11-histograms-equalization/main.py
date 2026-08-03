from src.image_loader import load_image
from src.preprocessing import preprocess_image

from src.grayscale_histogram import grayscale_histogram
from src.color_histogram import color_histogram

from src.equalization import histogram_equalization
from src.clahe import apply_clahe
from src.comparison import compare_images

from src.save import save_image
from src.utils import print_image_info


def main():

    print("=" * 70)
    print("Project 11 : Histograms & Histogram Equalization")
    print("=" * 70)

    image = load_image("images/sample.jpg")

    print_image_info(image)

    gray = preprocess_image(image)

    # Histogram Analysis
    gray_hist = grayscale_histogram(gray)
    color_hist = color_histogram(image)

    print(f"\n[INFO] Gray Histogram Shape : {gray_hist.shape}")

    for channel in ("blue", "green", "red"):
        print(
            f"[INFO] {channel.capitalize()} Histogram Shape : "
            f"{color_hist[channel].shape}"
        )

    # Histogram Equalization
    equalized = histogram_equalization(gray)

    # CLAHE
    clahe = apply_clahe(gray)

    # Image Comparison
    compare_images(
        ("Original", gray),
        ("Equalized", equalized),
        ("CLAHE", clahe)
    )

    # Save Images
    save_image(gray, "gray.jpg")
    save_image(equalized, "equalized.jpg")
    save_image(clahe, "clahe.jpg")

    print("\n" + "=" * 70)
    print("Project 11 Completed Successfully!")
    print("Results saved in output/")
    print("=" * 70)


if __name__ == "__main__":
    main()
