import numpy as np
import cv2

VIDEO_PATH = 'video/video8.MP4'
AREA_THRESHOLD = 900
PERIMETER_THRESHOLD = 200
MARKER_SIZE_MM = 19

WINDOW_NAME = 'Contours'
WINDOW_SIZE = (800, 600)

LOWER_RED1 = np.array([0, 100, 100])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([160, 100, 100])
UPPER_RED2 = np.array([180, 255, 255])


def setup_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *WINDOW_SIZE)


