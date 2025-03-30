import cv2
import numpy as np
from scale_converter import ScaleConverter
from width_tracker import WidthTracker
from detection import detect_red_squares_watershed, get_contour_edges_at_y
from segmentation import segment_person_kmeans
from config import WINDOW_SIZE, VIDEO_PATH


def main():
    scale_converter = ScaleConverter()
    width_tracker = WidthTracker()
    cap = cv2.VideoCapture(VIDEO_PATH)

    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', *WINDOW_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Сегментация человека
        person_mask = segment_person_kmeans(frame)
        contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            cv2.imshow("Result", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        person_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, [person_contour], -1, (255, 0, 0), 2)

        # 2. Обнаружение красных квадратов
        red_squares = detect_red_squares_watershed(frame)
        body_squares = []

        for (x, y, w, h) in red_squares:
            cX, cY = x + w // 2, y + h // 2
            if cv2.pointPolygonTest(person_contour, (cX, cY), False) >= 0:
                body_squares.append((x, y, w, h, cX, cY))

        if body_squares:
            median_size = np.median([max(w, h) for (x, y, w, h, cX, cY) in body_squares])
            scale_converter.update_scale(median_size)

        body_squares.sort(key=lambda sq: sq[4])
        current_widths = [None, None, None]

        # 3. Обработка квадратов и измерение ширины
        for i, (x, y, w, h, cX, cY) in enumerate(body_squares[:3]):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if scale_converter.pixels_per_mm:
                w_mm = scale_converter.px_to_mm(w)
                h_mm = scale_converter.px_to_mm(h)
                cv2.putText(frame, f"{i + 1}: {w_mm:.1f}x{h_mm:.1f}mm",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            left, right = get_contour_edges_at_y(person_contour, cY, frame.shape)
            if left and right:
                cv2.line(frame, (left, cY), (right, cY), (0, 255, 255), 2)
                if scale_converter.pixels_per_mm:
                    line_width_mm = scale_converter.px_to_mm(right - left)
                    current_widths[i] = line_width_mm
                    cv2.putText(frame, f"{line_width_mm:.1f}mm",
                                ((left + right) // 2, cY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        width_tracker.update(current_widths)
        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    width_tracker.plot()


if __name__ == "__main__":
    main()