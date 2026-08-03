from src.image_loader import load_image
from src.harris import harris_corner_detection
from src.shi_tomasi import shi_tomasi_detection
from src.fast import fast_feature_detection
from src.orb import orb_feature_detection
from src.sift import sift_feature_detection
from src.feature_matching import match_features
from src.save import save_image
from src.utils import print_feature_statistics


def main():

    print("=" * 60)
    print("Project 8 : Feature Detection & Description")
    print("=" * 60)

    image = load_image("images/sample.jpg")

    # Harris
    harris = harris_corner_detection(image)
    save_image(harris, "harris.jpg")

    # Shi-Tomasi
    shi = shi_tomasi_detection(image)
    save_image(shi, "shi_tomasi.jpg")

    # FAST
    fast, kp_fast = fast_feature_detection(image)
    save_image(fast, "fast.jpg")

    # ORB
    orb, kp_orb, des_orb = orb_feature_detection(image)
    save_image(orb, "orb.jpg")
    print_feature_statistics(
        "ORB",
        kp_orb,
        des_orb
    )

    # SIFT
    sift, kp_sift, des_sift = sift_feature_detection(image)

    if sift is not None:
        save_image(sift, "sift.jpg")
        print_feature_statistics(
            "SIFT",
            kp_sift,
            des_sift
        )

    # ORB Matching
    matched = match_features(
        image,
        kp_orb,
        des_orb,
        image,
        kp_orb,
        des_orb,
        method="ORB",
        max_matches=50
    )

    save_image(
        matched,
        "orb_matches.jpg"
    )

    print("\n" + "=" * 60)
    print("Project 8 Completed Successfully!")
    print("Results saved in output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
