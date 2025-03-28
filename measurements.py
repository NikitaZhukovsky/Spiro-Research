import cv2


def process_contours(skin_contours, red_contours, area_threshold, perimeter_threshold):
    red_centers = []
    valid_skin_contours = []

    for cnt in skin_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > area_threshold and perimeter > perimeter_threshold:
            valid_skin_contours.append(cnt)

            for red_cnt in red_contours:
                M = cv2.moments(red_cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if cv2.pointPolygonTest(cnt, (cx, cy), False) >= 0:
                        red_centers.append((cx, cy))

    return valid_skin_contours, red_centers


def calculate_pixel_to_mm_ratio(red_contour, marker_size_mm):
    if red_contour:
        x, y, w, h = cv2.boundingRect(red_contour[0])
        return marker_size_mm / w
    return None
