"""
Лабораторные работы 3 и 4 (поверх приложения первой лабораторной, task1.py).

Лабораторная 3. Замена цветного RGB-изображения (Bitmap, 24 бита на пиксель) на
изображение в градациях серого (тоже Bitmap, 24 бита на пиксель).
Пиксель в градациях серого - это пиксель, у которого R = G = B = Y, где Y - яркость.
Если на изображении выделена зона интереса, преобразуется только она, иначе - всё
изображение. Операция отменяется (Ctrl+Z).

Лабораторная 4. Вычисление минимальной m и максимальной M амплитуд, выборочного
среднего x и стандартного отклонения s амплитуд пикселей зоны интереса D: отдельно
для всей области и только для её границы. Для цветного (векторного) изображения
считается по каждой компоненте R, G, B.
    x = (1/|D|) * sum(x_a),   s^2 = (1/|D|) * sum((x_a - x)^2).

Файл расширяет приложение первой лабораторной (task1.py): положите его рядом.
Запуск:  python task3.py
"""

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from task1 import ImageApp, Roi, arrays_to_image, image_to_arrays

# Веса ITU-R BT.601 в тысячных долях: Y = 0.299 R + 0.587 G + 0.114 B.
# Глаз сильнее всего чувствителен к зелёному и слабее всего к синему, поэтому вклад
# компонент в яркость разный.
WEIGHT_R, WEIGHT_G, WEIGHT_B = 299, 587, 114

METHODS = {
    "mean": "среднее арифметическое",
}


# def gray_weighted(r, g, b):
#     """Яркость как взвешенная сумма компонент. Все вычисления целочисленные."""
#     s = (r.astype(np.uint32) * WEIGHT_R
#          + g.astype(np.uint32) * WEIGHT_G
#          + b.astype(np.uint32) * WEIGHT_B)
#     return ((s + 500) // 1000).astype(np.uint8)       # +500: округление к ближайшему целому


def gray_mean(r, g, b):
    """Яркость как среднее арифметическое компонент."""
    s = r.astype(np.uint16) + g + b                    # максимум 765, в uint16 помещается
    return ((s + 1) // 3).astype(np.uint8)             # округление к ближайшему целому


GRAY_FUNCTIONS = {"mean": gray_mean}


def to_gray_image(img, roi=None, method="mean"):
    """Возвращает новое 24-битное RGB-изображение в градациях серого.

    Если задана зона интереса roi, в серое переводятся только пиксели внутри неё,
    остальные не меняются.
    """
    r, g, b = image_to_arrays(img)                     # Bitmap -> три массива byte
    y = GRAY_FUNCTIONS[method](r, g, b)                # яркость каждого пикселя
    if roi is None:
        return arrays_to_image(y, y, y)                # R = G = B = Y
    mask = roi.full_mask(r.shape)                      # True внутри зоны
    return arrays_to_image(np.where(mask, y, r),
                           np.where(mask, y, g),
                           np.where(mask, y, b))       # три массива byte -> Bitmap


# ---------------------------------------------------------------- лабораторная 4
def boundary_mask(mask):
    """Граница зоны: пиксели зоны, у которых хотя бы один из 4 соседей не входит в зону.

    Для прямоугольника это кольцо шириной в 1 пиксель по его периметру
    (2w + 2h - 4 пикселей); если одна из сторон равна 1, граница - вся зона.
    Соседи за пределами зоны (в том числе за краем изображения) считаются внешними.
    """
    p = np.pad(mask, 1, constant_values=False)
    inner = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1]
             & p[1:-1, :-2] & p[1:-1, 2:])                 # пиксель и все 4 соседа в зоне
    return mask & ~inner


def amplitude_stats(values):
    """Минимум m, максимум M, среднее x и СКО s набора амплитуд (одномерный массив).

    s считается по формуле из задания с делителем |D| (не |D| - 1):
    s^2 = (1/|D|) * sum((x_a - x)^2). Среднее вычитается до возведения в квадрат
    (так вычисление точнее, чем через E[x^2] - x^2).
    """
    x = values.astype(np.float64)
    mean = x.mean()
    s = float(np.sqrt(((x - mean) ** 2).mean()))
    return int(values.min()), int(values.max()), float(mean), s


def zone_statistics(img, roi=None):
    """Статистика амплитуд по зоне roi (если зоны нет, D = A, всё изображение).

    Возвращает (зона, результат): результат["area"] и результат["border"] - словари
    {"R": (m, M, x, s), "G": ..., "B": ..., "count": |D|} для всей области и для границы.
    """
    arrays = image_to_arrays(img)                      # три компоненты: R, G, B
    if roi is None:
        h, w = arrays[0].shape
        roi = Roi.rect(0, 0, w, h)
    result = {}
    for key, mask in (("area", roi.mask), ("border", boundary_mask(roi.mask))):
        stats = {name: amplitude_stats(roi.crop(arr)[mask])
                 for name, arr in zip("RGB", arrays)}
        stats["count"] = int(mask.sum())
        result[key] = stats
    return roi, result


def _submenu(menubar, label):
    """Выпадающее меню по подписи (если такого нет - создаётся новое)."""
    for i in range(menubar.index("end") + 1):
        if menubar.type(i) == "cascade" and menubar.entrycget(i, "label") == label:
            return menubar.nametowidget(menubar.entrycget(i, "menu"))
    menu = tk.Menu(menubar, tearoff=False)
    menubar.add_cascade(label=label, menu=menu)
    return menu


class GrayscaleApp(ImageApp):
    """Приложение первой лабораторной + градации серого (лаб. 3) + статистика зоны (лаб. 4)."""

    def __init__(self):
        super().__init__()
        self.gray_method = tk.StringVar(value="mean")

        self._stats_window = None                      # окно со статистикой (лаб. 4)

        # добавляем пункты в уже существующие меню
        menubar = self.nametowidget(self.cget("menu"))
        image_menu = _submenu(menubar, "Изображение")
        image_menu.add_separator()
        image_menu.add_command(label="В градации серого", accelerator="Ctrl+G",
                               command=self.to_grayscale)
        for value, text in METHODS.items():
            image_menu.add_radiobutton(label=f"{text}",
                                       variable=self.gray_method, value=value)
        self.bind("<Control-g>", lambda e: self.to_grayscale())

        roi_menu = _submenu(menubar, "Зона интереса")
        roi_menu.add_separator()
        roi_menu.add_command(label="Статистика зоны (min, max, среднее, СКО)...",
                             accelerator="Ctrl+I", command=self.show_zone_statistics)
        self.bind("<Control-i>", lambda e: self.show_zone_statistics())

    def to_grayscale(self):
        """Заменить цветное изображение (или зону интереса) на серое."""
        if self.image is None:
            messagebox.showinfo("Градации серого", "Сначала откройте изображение.")
            return
        roi = self.roi
        new = to_gray_image(self.image, roi, self.gray_method.get())
        if np.array_equal(np.asarray(new), np.asarray(self.image)):
            messagebox.showinfo("Градации серого",
                                "Зона уже в градациях серого." if roi is not None
                                else "Изображение уже в градациях серого.")
            return
        self.push_undo("преобразование в градации серого"
                       + (" (в зоне)" if roi is not None else ""))
        x, y = self.canvas.xview()[0], self.canvas.yview()[0]
        self._display(new)                             # зона остаётся: размер изображения прежний
        self.canvas.xview_moveto(x)                    # прокрутка не сбрасывается
        self.canvas.yview_moveto(y)
        self._update_status()

    # ------------------------------------------------------------ лабораторная 4
    def show_zone_statistics(self):
        """Посчитать m, M, x, s по зоне интереса и показать в окне (область и граница)."""
        if self.image is None:
            messagebox.showinfo("Статистика зоны", "Сначала откройте изображение.")
            return
        roi, result = zone_statistics(self.image, self.roi)

        if self._stats_window is not None and self._stats_window.winfo_exists():
            self._stats_window.destroy()               # одно окно: старое заменяем новым
        win = self._stats_window = tk.Toplevel(self)
        win.title("Статистика зоны интереса")
        win.transient(self)
        win.resizable(False, False)
        body = tk.Frame(win, padx=12, pady=10)
        body.pack()

        if self.roi is None:
            zone = f"Зона не выбрана: D = A, всё изображение ({roi.width} x {roi.height})"
        else:
            kind = "многоугольник" if roi.kind == "polygon" else "прямоугольник"
            zone = f"Зона D ({kind}): x={roi.x0}, y={roi.y0}, {roi.width} x {roi.height}"
        tk.Label(body, text=zone, anchor="w").pack(fill="x", pady=(0, 8))

        columns = (("name", "Компонента", 110), ("min", "min  m", 70), ("max", "max  M", 70),
                   ("mean", "среднее  x\u0304", 110), ("std", "СКО  s", 110))
        for key, title in (("area", "Вся область D"), ("border", "Только граница D")):
            stats = result[key]
            frame = tk.LabelFrame(body, text=f"{title}   |D| = {stats['count']} пикс.",
                                  padx=8, pady=6)
            frame.pack(fill="x", pady=4)
            table = ttk.Treeview(frame, columns=[c[0] for c in columns],
                                 show="headings", height=3, selectmode="none")
            for name, text, width in columns:
                table.heading(name, text=text)
                table.column(name, width=width, anchor="w" if name == "name" else "e")
            for name in "RGB":
                m, big_m, mean, s = stats[name]
                table.insert("", "end", values=(name, m, big_m, f"{mean:.4f}", f"{s:.4f}"))
            table.pack()


        tk.Button(body, text="Закрыть", width=10, command=win.destroy).pack(pady=(8, 0))
        win.bind("<Escape>", lambda e: win.destroy())
        win.focus_set()                                # Esc работает сразу, без клика по окну


if __name__ == "__main__":
    GrayscaleApp().mainloop()