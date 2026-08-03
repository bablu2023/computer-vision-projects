from src.image_loader import load_image
from src.image_display import display_image
from src.line import draw_line
from src.rectangle import draw_rectangle
from src.circle import draw_circle
from src.ellipse import draw_ellipse
from src.polygon import draw_polygon
from src.arrow import draw_arrow
from src.text import draw_text
from src.annotation import annotate_image
from src.save import save_image
from src.utils import print_image_info


def main():
    print("=" * 60)
    print("Project 5 : Drawing and Annotation using OpenCV")
    print("=" * 60)

    # --------------------------------------------------
    # Load Image
    # --------------------------------------------------
    image = load_image("images/sample.jpg")
    print_image_info(image)

    display_image(image, "Original Image")

    # --------------------------------------------------
    # Line
    # --------------------------------------------------
    line_img = draw_line(image)
    display_image(line_img, "Line")
    save_image(line_img, "line.jpg")

    # --------------------------------------------------
    # Rectangle
    # --------------------------------------------------
    rectangle_img = draw_rectangle(image)
    display_image(rectangle_img, "Rectangle")
    save_image(rectangle_img, "rectangle.jpg")

    # --------------------------------------------------
    # Circle
    # --------------------------------------------------
    circle_img = draw_circle(image)
    display_image(circle_img, "Circle")
    save_image(circle_img, "circle.jpg")

    # --------------------------------------------------
    # Ellipse
    # --------------------------------------------------
    ellipse_img = draw_ellipse(image)
    display_image(ellipse_img, "Ellipse")
    save_image(ellipse_img, "ellipse.jpg")

    # --------------------------------------------------
    # Polygon
    # --------------------------------------------------
    polygon_img = draw_polygon(image)
    display_image(polygon_img, "Polygon")
    save_image(polygon_img, "polygon.jpg")

    # --------------------------------------------------
    # Arrow
    # --------------------------------------------------
    arrow_img = draw_arrow(image)
    display_image(arrow_img, "Arrow")
    save_image(arrow_img, "arrow.jpg")

    # --------------------------------------------------
    # Text
    # --------------------------------------------------
    text_img = draw_text(
        image,
        text="Hello OpenCV!",
        position=(40, 60),
        color=(0, 255, 0)
    )

    display_image(text_img, "Text")
    save_image(text_img, "text.jpg")

    # --------------------------------------------------
    # Combined Annotation
    # --------------------------------------------------
    annotated_img = annotate_image(image)

    display_image(annotated_img, "Annotated Image")
    save_image(annotated_img, "annotated.jpg")

    print("\n" + "=" * 60)
    print("Project Completed Successfully!")
    print("All output images are saved in the output/ directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
