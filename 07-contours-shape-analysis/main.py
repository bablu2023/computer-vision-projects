from src.image_loader import load_image
from src.preprocessing import preprocess_image
from src.contours import find_contours
from src.draw_contours import draw_all_contours
from src.area_perimeter import contour_properties
from src.bounding_box import draw_bounding_boxes
from src.min_circle import draw_min_enclosing_circles
from src.polygon import draw_polygons
from src.convex_hull import draw_convex_hulls
from src.moments import draw_centroids
from src.shape_detector import detect_shapes
from src.save import save_image
from src.utils import (
    print_image_info,
    filter_contours,
)


def main():

    print("=" * 60)
    print("Project 7 : Contours & Shape Analysis")
    print("=" * 60)

    # --------------------------------------------------
    # Load Image
    # --------------------------------------------------
    image = load_image("images/sample.jpg")
    print_image_info(image)

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    gray, blurred, binary = preprocess_image(image)

    save_image(gray, "gray.jpg")
    save_image(blurred, "blurred.jpg")
    save_image(binary, "binary.jpg")

    # --------------------------------------------------
    # Find Contours
    # --------------------------------------------------
    contours, hierarchy = find_contours(binary)

    contours = filter_contours(
        contours,
        min_area=100
    )

    print(f"[INFO] Valid contours: {len(contours)}")

    # --------------------------------------------------
    # Draw Contours
    # --------------------------------------------------
    contour_img = draw_all_contours(
        image,
        contours
    )

    save_image(
        contour_img,
        "contours.jpg"
    )

    # --------------------------------------------------
    # Area & Perimeter
    # --------------------------------------------------
    contour_properties(contours)

    # --------------------------------------------------
    # Bounding Boxes
    # --------------------------------------------------
    bbox_img = draw_bounding_boxes(
        image,
        contours,
        min_area=100
    )

    save_image(
        bbox_img,
        "bounding_boxes.jpg"
    )

    # --------------------------------------------------
    # Minimum Enclosing Circles
    # --------------------------------------------------
    circle_img = draw_min_enclosing_circles(
        image,
        contours,
        min_area=100
    )

    save_image(
        circle_img,
        "min_circles.jpg"
    )

    # --------------------------------------------------
    # Polygon Approximation
    # --------------------------------------------------
    polygon_img = draw_polygons(
        image,
        contours,
        min_area=100
    )

    save_image(
        polygon_img,
        "polygons.jpg"
    )

    # --------------------------------------------------
    # Convex Hulls
    # --------------------------------------------------
    hull_img = draw_convex_hulls(
        image,
        contours,
        min_area=100
    )

    save_image(
        hull_img,
        "convex_hulls.jpg"
    )

    # --------------------------------------------------
    # Centroids
    # --------------------------------------------------
    centroid_img = draw_centroids(
        image,
        contours,
        min_area=100
    )

    save_image(
        centroid_img,
        "centroids.jpg"
    )

    # --------------------------------------------------
    # Shape Detection
    # --------------------------------------------------
    shape_img = detect_shapes(
        image,
        contours,
        min_area=100
    )

    save_image(
        shape_img,
        "shapes.jpg"
    )

    print("\n" + "=" * 60)
    print("Project 7 Completed Successfully!")
    print("Results saved in output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
