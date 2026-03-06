import cv2
import cv2.aruco as aruco
import numpy as np

def generate_charuco_board():
    # 5x7 squares, 30mm square size, 22mm marker size
    # Dictionary: DICT_6X6_250
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    
    # 7 cols, 5 rows of squares
    # Squares: 300 pixels (representing 30mm)
    # Markers: 220 pixels (representing 22mm)
    board = aruco.CharucoBoard((7, 5), 0.030, 0.022, aruco_dict)
    
    # Generate image (7 squares * 300px = 2100px width; 5 * 300px = 1500px height) + 100px margins
    img = board.generateImage((2100 + 200, 1500 + 200), marginSize=100)
    
    cv2.imwrite("charuco_board_7x5_dict6x6.png", img)
    print("Saved charuco_board_7x5_dict6x6.png")

def generate_single_markers():
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    
    # Generate a sheet of 4 markers
    sheet = np.ones((2000, 2000), dtype=np.uint8) * 255
    
    # Draw 4 markers in corners
    m1 = aruco.generateImageMarker(aruco_dict, 0, 800)
    m2 = aruco.generateImageMarker(aruco_dict, 1, 800)
    m3 = aruco.generateImageMarker(aruco_dict, 2, 800)
    m4 = aruco.generateImageMarker(aruco_dict, 3, 800)
    
    sheet[100:900, 100:900] = m1
    sheet[100:900, 1100:1900] = m2
    sheet[1100:1900, 100:900] = m3
    sheet[1100:1900, 1100:1900] = m4
    
    cv2.imwrite("aruco_markers_0_to_3.png", sheet)
    print("Saved aruco_markers_0_to_3.png")

if __name__ == "__main__":
    generate_charuco_board()
    generate_single_markers()
