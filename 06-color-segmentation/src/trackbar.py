import cv2
import numpy as np


def nothing(x):
    """Dummy callback required by OpenCV trackbars."""
    pass


def hsv_trackbar(image):
    """
    Interactive HSV threshold tuning.

    Press ESC to exit.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.namedWindow("HSV Trackbar")

    # Lower HSV
    cv2.createTrackbar("H Min", "HSV Trackbar", 0, 179, nothing)
    cv2.createTrackbar("S Min", "HSV Trackbar", 0, 255, nothing)
    cv2.createTrackbar("V Min", "HSV Trackbar", 0, 255, nothing)

    # Upper HSV
    cv2.createTrackbar("H Max", "HSV Trackbar", 179, 179, nothing)
    cv2.createTrackbar("S Max", "HSV Trackbar", 255, 255, nothing)
    cv2.createTrackbar("V Max", "HSV Trackbar", 255, 255, nothing)

    print("[INFO] Press ESC to exit.")
    print("[INFO] Adjust the sliders to find the desired HSV range.")

    while True:

        h_min = cv2.getTrackbarPos("H Min", "HSV Trackbar")
        s_min = cv2.getTrackbarPos("S Min", "HSV Trackbar")
        v_min = cv2.getTrackbarPos("V Min", "HSV Trackbar")

        h_max = cv2.getTrackbarPos("H Max", "HSV Trackbar")
        s_max = cv2.getTrackbarPos("S Max", "HSV Trackbar")
        v_max = cv2.getTrackbarPos("V Max", "HSV Trackbar")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        result = cv2.bitwise_and(
            image,
            image,
            mask=mask
        )

        cv2.imshow("Original", image)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # ESC key
            break

    cv2.destroyAllWindows()

    print("[INFO] HSV Trackbar closed.")
