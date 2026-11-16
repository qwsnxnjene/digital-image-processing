"""
Сегментация изображения: отделение объекта от фона в зоне интереса.

Зону интереса выделяют вокруг объекта, поэтому её граница лежит на фоне. По яркости
пикселей границы считаются среднее x и СКО s (как в лабораторной со статистикой), затем
пользователь вводит коэффициент k, и интервал яркости фона получается таким:
    x - k*s  <=  яркость  <=  x + k*s
Пиксель зоны с яркостью внутри интервала - фон (белый, 255), вне интервала - объект
(чёрный, 0). Вне зоны изображение не меняется. Операция отменяется (Ctrl+Z).
Яркость считается как в градациях серого (взвешенная сумма R, G, B).

Файл расширяет приложение из task3.py: положите его рядом с task1.py и task3.py.
Запуск:  python task4.py
"""

from tkinter import messagebox, simpledialog

import numpy as np

from task1 import arrays_to_image, image_to_arrays
from task3 import GrayscaleApp, _submenu, amplitude_stats, boundary_mask, gray_weighted


def border_stats(img, roi):
    """Яркость каждого пикселя, а также среднее x и СКО s яркости по границе зоны (фон)."""
    r, g, b = image_to_arrays(img)
    y = gray_weighted(r, g, b)                         # яркость каждого пикселя изображения
    border = boundary_mask(roi.mask)                   # граница зоны: считаем её фоном
    _, _, mean, s = amplitude_stats(roi.crop(y)[border])
    return y, mean, s


def segment(img, roi, y, mean, s, k):
    """Новое изображение: в зоне фон белый, объект чёрный, вне зоны без изменений."""
    low, high = mean - k * s, mean + k * s             # интервал яркости фона
    background = (y >= low) & (y <= high)              # True - пиксель похож на фон
    value = np.where(background, 255, 0).astype(np.uint8)
    mask = roi.full_mask(y.shape)                      # True внутри зоны
    r, g, b = image_to_arrays(img)
    return arrays_to_image(np.where(mask, value, r),
                           np.where(mask, value, g),
                           np.where(mask, value, b))   # внутри зоны R = G = B


class SegmentationApp(GrayscaleApp):
    """Приложение из task3.py + сегментация зоны интереса."""

    def __init__(self):
        super().__init__()
        menubar = self.nametowidget(self.cget("menu"))
        roi_menu = _submenu(menubar, "Зона интереса")
        roi_menu.add_command(label="Сегментация зоны (объект / фон)...", accelerator="Ctrl+K",
                             command=self.segment_zone)
        self.bind("<Control-k>", lambda e: self.segment_zone())

    def segment_zone(self):
        """Отделить объект от фона в зоне интереса (интервал яркости фона x +- k*s)."""
        if self.image is None:
            messagebox.showinfo("Сегментация", "Сначала откройте изображение.")
            return
        if self.roi is None:
            messagebox.showinfo("Сегментация", "Сначала выделите зону интереса.")
            return
        y, mean, s = border_stats(self.image, self.roi)
        k = simpledialog.askfloat(
            "Сегментация",
            f"Фон оценивается по границе зоны:\n"
            f"x\u0304 = {mean:.2f},  s = {s:.2f}\n\n"
            f"Фон: x\u0304 \u2212 k\u00b7s \u2264 яркость \u2264 x\u0304 + k\u00b7s\n"
            f"Всё остальное - объект (чёрный).\n\n"
            f"Введите коэффициент k:",
            parent=self, initialvalue=3.0, minvalue=0.1)
        if k is None:                                  # диалог закрыт
            return
        new = segment(self.image, self.roi, y, mean, s, k)
        self.push_undo("сегментация зоны")
        x0, y0 = self.canvas.xview()[0], self.canvas.yview()[0]
        self._display(new)                             # зона остаётся на месте
        self.canvas.xview_moveto(x0)                   # прокрутка не сбрасывается
        self.canvas.yview_moveto(y0)
        self._update_status()


if __name__ == "__main__":
    SegmentationApp().mainloop()