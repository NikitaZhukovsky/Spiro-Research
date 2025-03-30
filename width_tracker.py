from collections import deque
import matplotlib.pyplot as plt
from config import PLOT_SIZE


class WidthTracker:
    def __init__(self, num_lines=3, max_history=100):
        self.width_history = [deque(maxlen=max_history) for _ in range(num_lines)]

    def update(self, widths_mm):
        for i, width in enumerate(widths_mm):
            if width is not None:
                self.width_history[i].append(width)

    def plot(self):
        for i, history in enumerate(self.width_history):
            if not history:
                continue

            plt.figure(figsize=PLOT_SIZE)
            plt.plot(history, linewidth=2, label=f'Линия {i + 1}')
            plt.xlabel('Номер кадра', fontsize=12)
            plt.ylabel('Ширина (мм)', fontsize=12)
            plt.title(f'Динамика изменения ширины тела по линии {i + 1}', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tick_params(axis='both', which='major', labelsize=10)

            y_min, y_max = min(history), max(history)
            y_range = y_max - y_min
            plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            plt.show()