import cv2
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────
SQUARES_X    = 7       # number of squares horizontally
SQUARES_Y    = 5       # number of squares vertically
SQUARE_SIZE  = 40      # size of each square in mm
MARKER_SIZE  = 30      # size of ArUco marker inside each square in mm
DICT         = cv2.aruco.DICT_6X6_50  # same dictionary as your floor tag

# Image size - A4 at 300 DPI is roughly 2480 x 3508 pixels
IMAGE_SIZE_MM  = (297, 210)  # A4 landscape in mm (width, height)
DPI            = 300
MM_TO_PIXELS   = DPI / 25.4  # 1mm in pixels at 300 DPI

image_width  = int(IMAGE_SIZE_MM[0] * MM_TO_PIXELS)
image_height = int(IMAGE_SIZE_MM[1] * MM_TO_PIXELS)

# ── GENERATE BOARD ────────────────────────────────────────────────
dictionary = cv2.aruco.getPredefinedDictionary(DICT)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_SIZE / 1000,   # convert mm to metres
    MARKER_SIZE / 1000,   # convert mm to metres
    dictionary
)

# Draw the board onto an image
board_image = board.generateImage(
    (image_width, image_height),
    marginSize=int(20 * MM_TO_PIXELS),  # 20mm white border
    borderBits=1
)

# Save it
output_path = "data/charuco_board.png"
cv2.imwrite(output_path, board_image)
print(f"ChArUco board saved to {output_path}")
print()
print("PRINT INSTRUCTIONS:")
print("  - Print on A4 paper, landscape orientation")
print("  - Set print scale to EXACTLY 100% (no fit to page)")
print("  - Do NOT scale to fit")
print()
print("AFTER PRINTING - measure and confirm:")
print(f"  - Each square should be exactly {SQUARE_SIZE}mm")
print(f"  - Each ArUco marker inside should be exactly {MARKER_SIZE}mm")
print(f"  - Total board width should be {SQUARES_X * SQUARE_SIZE}mm = {SQUARES_X * SQUARE_SIZE / 10}cm")
print(f"  - Total board height should be {SQUARES_Y * SQUARE_SIZE}mm = {SQUARES_Y * SQUARE_SIZE / 10}cm")
print()
print("Update CHARUCO_SQUARE_SIZE and CHARUCO_MARKER_SIZE in")
print("07_hand_eye_calibration.py with your measured values in metres")

# Show it on screen
cv2.imshow("ChArUco Board - Print This", board_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

