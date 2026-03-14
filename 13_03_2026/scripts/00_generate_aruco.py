import cv2
import numpy as np

# We are using the 4x4 ArUco dictionary
# This means tags that are a 4x4 grid of black and white squares
# ID 0 just means it's the first tag in the dictionary
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Generate the tag
# 500 means the image will be 500x500 pixels
tag_image = np.zeros((500, 500), dtype=np.uint8)
tag_image = cv2.aruco.generateImageMarker(dictionary, id=0, sidePixels=500)

# Add a white border around it so it's easier to detect
tag_with_border = cv2.copyMakeBorder(
    tag_image, 50, 50, 50, 50,
    cv2.BORDER_CONSTANT, value=255
)

# Save it
output_path = "data/aruco_tag_id0.png"
cv2.imwrite(output_path, tag_with_border)
print(f"ArUco tag saved to {output_path}")
print("Print it out and measure the black square size in metres before we continue")

# Also show it on screen
cv2.imshow("ArUco Tag - Print This", tag_with_border)
cv2.waitKey(0)
cv2.destroyAllWindows()

