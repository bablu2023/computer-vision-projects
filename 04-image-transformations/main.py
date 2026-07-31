import cv2

from src.image_loader import load_image
from src.image_display import display_image
from src.translation import translate_image
from src.scaling import scale_image
from src.rotation import rotate_image
from src.affine import affine_transform
from src.perspective import perspective_transform
from src.pyramid import gaussian_pyramid, laplacian_pyramid
from src.resize import resize_image
from src.save import save_image
from src.utils import print_image_info


def main():
    print("=" * 60)
    print("Project 4 : Image Transformations using OpenCV")
    print("=" * 60)

    # --------------------------------------------------
    # Load Image
    # --------------------------------------------------
    image = load_image("images/sample.jpg")

    print_image_info(image)

    display_image(image, "Original Image")

    # --------------------------------------------------
    # Translation
    # --------------------------------------------------
    translated = translate_image(image, tx=100, ty=50)
    display_image(translated, "Translated Image")
    save_image(translated, "translated.jpg")

    # --------------------------------------------------
    # Scaling
    # --------------------------------------------------
    scaled = scale_image(
        image,
        scale_x=1.5,
        scale_y=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    display_image(scaled, "Scaled Image")
    save_image(scaled, "scaled.jpg")

    # --------------------------------------------------
    # Rotation
    # --------------------------------------------------
    rotated = rotate_image(
        image,
        angle=45,
        scale=1.0
    )

    display_image(rotated, "Rotated Image")
    save_image(rotated, "rotated.jpg")

    # --------------------------------------------------
    # Affine Transformation
    # --------------------------------------------------
    affine = affine_transform(image)

    display_image(affine, "Affine Transformation")
    save_image(affine, "affine.jpg")

    # --------------------------------------------------
    # Perspective Transformation
    # --------------------------------------------------
    perspective = perspective_transform(image)

    display_image(
        perspective,
        "Perspective Transformation"
    )

    save_image(
        perspective,
        "perspective.jpg"
    )

    # --------------------------------------------------
    # Gaussian Pyramid
    # --------------------------------------------------
    gaussian = gaussian_pyramid(
        image,
        levels=3
    )

    for i, img in enumerate(gaussian):
        display_image(
            img,
            f"Gaussian Level {i}"
        )

        save_image(
            img,
            f"gaussian_level_{i}.jpg"
        )

    # --------------------------------------------------
    # Laplacian Pyramid
    # --------------------------------------------------
    laplacian = laplacian_pyramid(
        image,
        levels=3
    )

    for i, img in enumerate(laplacian):
        display_image(
            img,
            f"Laplacian Level {i}"
        )

        save_image(
            img,
            f"laplacian_level_{i}.jpg"
        )

    # --------------------------------------------------
    # Resize
    # --------------------------------------------------
    resized = resize_image(
        image,
        width=256,
        interpolation=cv2.INTER_AREA
    )

    display_image(
        resized,
        "Resized Image"
    )

    save_image(
        resized,
        "resized.jpg"
    )

    print("\n" + "=" * 60)
    print("Project Completed Successfully!")
    print("All transformed images have been saved in the output folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()
