from src.image_loader import load_image
from src.image_display import display_image

from src.blur import average_blur
from src.gaussian import gaussian_blur
from src.median import median_blur
from src.bilateral import bilateral_filter
from src.sharpen import sharpen_image
from src.edge import canny_edge
from src.threshold import apply_threshold
from src.histogram import plot_histogram, equalize_histogram
from src.save import save_image


def main():
    print("=" * 50)
    print("Project 3 - Image Filtering & Enhancement")
    print("=" * 50)

    # Load image
    image = load_image("images/sample.jpg")

    # Display original image
    display_image(image, "Original Image")

    # Plot original histogram
    plot_histogram(image, "Original Histogram")

    # Average Blur
    avg = average_blur(image)
    display_image(avg, "Average Blur")
    save_image(avg, "average_blur.jpg")

    # Gaussian Blur
    gaussian = gaussian_blur(image)
    display_image(gaussian, "Gaussian Blur")
    save_image(gaussian, "gaussian_blur.jpg")

    # Median Blur
    median = median_blur(image)
    display_image(median, "Median Blur")
    save_image(median, "median_blur.jpg")

    # Bilateral Filter
    bilateral = bilateral_filter(image)
    display_image(bilateral, "Bilateral Filter")
    save_image(bilateral, "bilateral_filter.jpg")

    # Sharpen
    sharpened = sharpen_image(image)
    display_image(sharpened, "Sharpened Image")
    save_image(sharpened, "sharpened.jpg")

    # Edge Detection
    edges = canny_edge(image)
    display_image(edges, "Canny Edge Detection", cmap="gray")
    save_image(edges, "edges.jpg")

    # Thresholding
    thresholds = apply_threshold(image)

    for name, result in thresholds.items():
        display_image(result, name.replace("_", " ").title(), cmap="gray")
        save_image(result, f"{name}.jpg")

    # Histogram Equalization
    equalized = equalize_histogram(image)
    display_image(equalized, "Histogram Equalization", cmap="gray")
    save_image(equalized, "equalized.jpg")

    # Plot Equalized Histogram
    plot_histogram(equalized, "Equalized Histogram")

    print("\n" + "=" * 50)
    print("Image Filtering & Enhancement Completed Successfully!")
    print("Processed images have been saved in the output/ directory.")
    print("=" * 50)


if __name__ == "__main__":
    main()
