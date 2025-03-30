import cv2
import numpy as np
from config import RED_LOWER1, RED_UPPER1, RED_LOWER2, RED_UPPER2


def detect_red_squares_watershed(frame):
    shifted = cv2.pyrMeanShiftFiltering(frame, 15, 30)
    hsv = cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask = mask1 | mask2

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue

        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = float(w_box) / float(h_box)

        if 0.8 < aspect_ratio < 1.2:
            squares.append((x, y, w_box, h_box))

    return squares


def get_contour_edges_at_y(contour, y_pos, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    if y_pos >= img_shape[0]:
        return None, None

    row = mask[y_pos, :]
    left = np.argmax(row > 0)
    right = len(row) - np.argmax(row[::-1] > 0) - 1

    if left >= right:
        return None, None

    return left, right