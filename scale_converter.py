from config import REFERENCE_SIZE_MM


class ScaleConverter:
    def __init__(self):
        self.pixels_per_mm = None
        self.reference_size_mm = REFERENCE_SIZE_MM

    def update_scale(self, square_width_px):
        if square_width_px > 0:
            self.pixels_per_mm = square_width_px / self.reference_size_mm
            print(f"Масштаб: {self.pixels_per_mm:.2f} px/mm")

    def px_to_mm(self, px):
        if self.pixels_per_mm is None:
            return None
        return px / self.pixels_per_mm

    def mm_to_px(self, mm):
        if self.pixels_per_mm is None:
            return None
        return mm * self.pixels_per_mm