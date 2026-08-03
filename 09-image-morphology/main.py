from src.image_loader import load_image
from src.preprocessing import preprocess_image
from src.kernel import create_kernel

from src.erosion import erode_image
from src.dilation import dilate_image
from src.opening import opening_image
from src.closing import closing_image
from src.gradient import gradient_image
from src.tophat import tophat_image
from src.blackhat import blackhat_image

from src.save import save_image
from src.utils import print_image_info


def main():

    print("=" * 60)
    print("Project 9 : Image Morphology")
    print("=" * 60)

    image = load_image("images/sample.jpg")

    print_image_info(image)

    gray, binary = preprocess_image(image)

    kernel = create_kernel(
        shape="rect",
        size=(5, 5)
    )

    eroded = erode_image(
        binary,
        kernel
    )

    dilated = dilate_image(
        binary,
        kernel
    )

    opened = opening_image(
        binary,
        kernel
    )

    closed = closing_image(
        binary,
        kernel
    )

    gradient = gradient_image(
        binary,
        kernel
    )

    tophat = tophat_image(
        binary,
        kernel
    )

    blackhat = blackhat_image(
        binary,
        kernel
    )

    save_image(gray, "gray.jpg")
    save_image(binary, "binary.jpg")
    save_image(eroded, "eroded.jpg")
    save_image(dilated, "dilated.jpg")
    save_image(opened, "opening.jpg")
    save_image(closed, "closing.jpg")
    save_image(gradient, "gradient.jpg")
    save_image(tophat, "tophat.jpg")
    save_image(blackhat, "blackhat.jpg")

    print("\n" + "=" * 60)
    print("Project 9 Completed Successfully!")
    print("Results saved in output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
