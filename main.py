from config import *
from masks import skin_mask, red_square_mask
from plotting import plot_measurements
import cv2
import numpy as np


setup_window()

cap = cv2.VideoCapture(VIDEO_PATH)

width_data_mm = {1: [], 2: [], 3: []}
current_lengths = {1: 0, 2: 0, 3: 0}  # Для хранения текущих длин

frame_count = 0
pixel_to_mm_ratio = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    skin = skin_mask(frame)
    red_square = red_square_mask(frame)

    skin_contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_square, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_frame = frame.copy()
    red_centers = []

    for cnt in skin_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > AREA_THRESHOLD and perimeter > PERIMETER_THRESHOLD:
            cv2.drawContours(contour_frame, [cnt], -1, (0, 255, 0), 2)

            for red_cnt in red_contours:
                M = cv2.moments(red_cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    if cv2.pointPolygonTest(cnt, (cx, cy), False) >= 0:
                        red_centers.append((cx, cy))
                        cv2.circle(contour_frame, (cx, cy), 5, (0, 0, 255), -1)
                        if pixel_to_mm_ratio is None:
                            x, y, w, h = cv2.boundingRect(red_cnt)
                            pixel_to_mm_ratio = MARKER_SIZE_MM / w

                        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        cv2.drawContours(mask, [cnt], -1, 255, -1)
                        horizontal_projection = np.where(mask[cy, :] > 0)[0]

                        if len(horizontal_projection) > 0:
                            width_pixels = horizontal_projection[-1] - horizontal_projection[0]
                            width_mm = width_pixels * pixel_to_mm_ratio

                            marker_index = len(red_centers)
                            if marker_index in width_data_mm:
                                width_data_mm[marker_index].append(width_mm)
                                current_lengths[marker_index] = width_mm

    if red_centers:
        print(f"Кадр {frame_count}: Центры красных меток внутри зеленых контуров:", red_centers)

    cv2.drawContours(contour_frame, red_contours, -1, (255, 0, 0), 2)

    for index, center in enumerate(red_centers):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        leftmost = np.min(np.where(mask[center[1], :] > 0)[0])
        rightmost = np.max(np.where(mask[center[1], :] > 0)[0])

        length_pixels = rightmost - leftmost
        length_mm = length_pixels * pixel_to_mm_ratio

        # Обновляем текущую длину
        current_lengths[index + 1] = length_mm

        # Рисуем линию
        cv2.line(contour_frame, (leftmost, center[1]), (rightmost, center[1]), (255, 255, 0), 1)

        # Добавляем текст с длиной рядом с линией
        text = f"{length_mm:.2f} mm"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        # Позиция текста - справа от линии
        text_x = rightmost + 10
        text_y = center[1] + text_size[1] // 2

        # Фон для текста (для лучшей читаемости)
        cv2.rectangle(contour_frame,
                      (text_x - 2, text_y - text_size[1] - 2),
                      (text_x + text_size[0] + 2, text_y + 2),
                      (0, 0, 0), -1)

        # Сам текст
        cv2.putText(contour_frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Номер метки
        cv2.putText(contour_frame, str(index + 1), (center[0] - 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Contours', contour_frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plot_measurements(width_data_mm)