from src.image_loader import load_image
from src.preprocessing import preprocess_image

from src.global_threshold import global_threshold
from src.adaptive_threshold import adaptive_threshold
from src.otsu_threshold import otsu_threshold

from src.sobel import sobel_edge_detection
from src.scharr import scharr_edge_detection
from src.laplacian import laplacian_edge_detection
from src.canny import canny_edge_detection

from src.save import save_image
from src.utils import print_image_info


def main():

    print("=" * 60)
    print("Project 10 : Image Thresholding & Edge Detection")
    print("=" * 60)

    image = load_image("images/sample.jpg")

    print_image_info(image)

    gray = preprocess_image(image)

    save_image(gray, "gray.jpg")

    # Global Thresholding
    global_results = global_threshold(gray)

    for name, img in global_results.items():
        save_image(img, f"{name}.jpg")

    # Adaptive Thresholding
    adaptive_results = adaptive_threshold(gray)

    for name, img in adaptive_results.items():
        save_image(img, f"{name}.jpg")

    # Otsu Thresholding
    threshold, otsu = otsu_threshold(gray)

    print(f"[INFO] Otsu Threshold = {threshold:.2f}")

    save_image(otsu, "otsu.jpg")

    # Sobel
    sobel_x, sobel_y, sobel = sobel_edge_detection(gray)

    save_image(sobel_x, "sobel_x.jpg")
    save_image(sobel_y, "sobel_y.jpg")
    save_image(sobel, "sobel.jpg")

    # Scharr
    scharr_x, scharr_y, scharr = scharr_edge_detection(gray)

    save_image(scharr_x, "scharr_x.jpg")
    save_image(scharr_y, "scharr_y.jpg")
    save_image(scharr, "scharr.jpg")

    # Laplacian
    laplacian = laplacian_edge_detection(gray)

    save_image(laplacian, "laplacian.jpg")

    # Canny
    canny = canny_edge_detection(gray)

    save_image(canny, "canny.jpg")

    print("\n" + "=" * 60)
    print("Project 10 Completed Successfully!")
    print("Results saved in output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
