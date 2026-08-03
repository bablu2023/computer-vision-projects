import cv2
import os

image = cv2.imread("images/sample.jpg")

if image is None:
    raise FileNotFoundError("images/sample.jpg not found")

# Crop a 100x100 template from the center
template = image[150:250, 150:250]

cv2.imwrite("images/template.jpg", template)

print("[INFO] Template created successfully.")
print("[INFO] Saved: images/template.jpg")
