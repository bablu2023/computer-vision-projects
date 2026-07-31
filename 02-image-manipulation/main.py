from src.image_loader import load_image
from src.image_display import display_image
from src.resize import resize_image
from src.crop import crop_image
from src.rotate import rotate_image
from src.flip import flip_image
from src.save import save_image


def main():
    # Input image
    image_path = "images/sample.jpg"

    # Load image
    image = load_image(image_path)

    # Display original
    display_image("Original Image", image)

    # Resize (50%)
    resized = resize_image(image, scale=0.5)
    save_image("output/resized.jpg", resized)
    display_image("Resized Image", resized)

    # Crop
    cropped = crop_image(resized, x=50, y=50, width=150, height=150)
    save_image("output/cropped.jpg", cropped)
    display_image("Cropped Image", cropped)

    # Rotate
    rotated = rotate_image(cropped, angle=45)
    save_image("output/rotated.jpg", rotated)
    display_image("Rotated Image", rotated)

    # Flip
    flipped = flip_image(rotated, flip_code=1)
    save_image("output/flipped.jpg", flipped)
    display_image("Flipped Image", flipped)

    print("\n===================================")
    print(" Image Manipulation Completed!")
    print(" Results saved in: output/")
    print("===================================")


if __name__ == "__main__":
    main()
