import colorsys
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image

from task1 import ImageApp


MODES = {
    "rect": "Прямоугольники (горизонтальные и вертикальные разрезы)",
    "free": "Произвольные фигуры (разрез под случайным углом)",
}
MAX_SIDE = 5000


class PartitionError(Exception):
    """Не удалось построить разбиение с заданными параметрами."""


def _connected(mask):
    """Проверка связности участка по строкам (4-связность)"""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0 or rows[-1] - rows[0] + 1 != rows.size:
        return False
    m = mask[rows[0]:rows[-1] + 1]
    left = m.argmax(axis=1)
    right = m.shape[1] - 1 - m[:, ::-1].argmax(axis=1)
    if np.any(right - left + 1 != m.sum(axis=1)):
        return False
    return bool(np.all(
        np.maximum(left[:-1], left[1:]) <= np.minimum(right[:-1], right[1:])
    ))


def _bbox(mask, y0, x0):
    """Ограничивающий прямоугольник маски"""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return (
        y0 + rows[0], y0 + rows[-1] + 1,
        x0 + cols[0], x0 + cols[-1] + 1,
    )


def random_partition(h, w, n, mode="rect", seed=None):
    """Возвращает карту участков (h, w) со значениями от 1 до n"""
    if h < 1 or w < 1:
        raise ValueError("Размер изображения должен быть не меньше 1 x 1.")
    if not 1 <= n <= h * w:
        raise ValueError(f"Число участков должно быть от 1 до {h * w}.")
    if mode not in MODES:
        raise ValueError(f"Неизвестный способ разбиения: {mode!r}.")

    rng = np.random.default_rng(seed)
    angles = (0, 90) if mode == "rect" else tuple(range(180))
    labels = np.ones((h, w), dtype=np.int32)
    boxes = {1: (0, h, 0, w)}
    areas = {1: h * w}
    failures = 0
    budget = 2000 * n + 200_000

    for new in range(2, n + 1):
        while True:
            px, py = int(rng.integers(w)), int(rng.integers(h))
            k = int(labels[py, px])
            if areas[k] >= 2:
                angle = math.radians(angles[int(rng.integers(len(angles)))])
                dx, dy = round(math.cos(angle), 12), round(math.sin(angle), 12)
                y0, y1, x0, x1 = boxes[k]
                part = labels[y0:y1, x0:x1]
                inside = part == k
                xs = np.arange(x0, x1)[None, :]
                ys = np.arange(y0, y1)[:, None]
                pos = inside & (((xs - px) * dy - (ys - py) * dx) > 0)
                neg = inside & ~pos
                if pos.any() and neg.any() and _connected(pos) and _connected(neg):
                    break
            failures += 1
            if failures > budget:
                raise PartitionError(
                    f"Не удалось разбить изображение {w} x {h} на {n} участков. "
                    "Попробуйте уменьшить число участков."
                )
        part[pos] = new
        boxes[k], boxes[new] = _bbox(neg, y0, x0), _bbox(pos, y0, x0)
        areas[new] = int(pos.sum())
        areas[k] -= areas[new]

    return labels


def labels_to_image(labels, seed=0):
    """Создаёт цветную карту: у каждого участка свой цвет."""
    n = int(labels.max())
    rng = np.random.default_rng(seed)
    palette = np.zeros((n + 1, 3), dtype=np.uint8)
    shift = rng.random()
    for i in range(1, n + 1):
        hue = (i * 0.6180339887498949 + shift) % 1.0
        r, g, b = colorsys.hsv_to_rgb(
            hue, rng.uniform(0.45, 0.85), rng.uniform(0.80, 1.0)
        )
        palette[i] = (int(r * 255), int(g * 255), int(b * 255))
    return Image.fromarray(palette[labels], "RGB")


def draw_borders(image, labels):
    """Накладывает тонкие тёмные границы на оригинальное изображение."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    # Затемняем пиксели границ, сохраняя видимой исходную фотографию.
    rgb[edge] = (rgb[edge].astype(np.float32) * 0.45).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


class PartitionDialog(tk.Toplevel):
    """Диалог параметров разбиения."""

    def __init__(self, parent, max_n):
        super().__init__(parent)
        self.title("Разбиение на участки")
        self.transient(parent)
        self.resizable(False, False)
        self.result = None
        self.max_n = max_n

        self.v_n = tk.StringVar(value="10")
        self.v_mode = tk.StringVar(value="rect")

        body = tk.Frame(self, padx=14, pady=12)
        body.pack()

        tk.Label(body, text=f"Число участков n (1–{max_n}):", anchor="w").grid(
            row=0, column=0, sticky="w", pady=2
        )
        entry = tk.Entry(body, textvariable=self.v_n, width=14)
        entry.grid(row=0, column=1, sticky="w", padx=(8, 0), pady=2)
        entry.focus_set()
        entry.select_range(0, "end")

        tk.Label(body, text="Способ разбиения:", anchor="w").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        for row, (value, label) in enumerate(MODES.items(), start=2):
            tk.Radiobutton(
                body, text=label, variable=self.v_mode, value=value
            ).grid(row=row, column=0, columnspan=2, sticky="w")

        buttons = tk.Frame(body)
        buttons.grid(row=4, column=0, columnspan=2, pady=(12, 0))
        tk.Button(buttons, text="Разбить", width=10, command=self._ok).pack(
            side="left", padx=4
        )
        tk.Button(buttons, text="Отмена", width=10, command=self.destroy).pack(
            side="left", padx=4
        )

        self.bind("<Return>", lambda event: self._ok())
        self.bind("<Escape>", lambda event: self.destroy())
        self.grab_set()

    def _ok(self):
        try:
            n = int(self.v_n.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Число участков должно быть целым числом.", parent=self)
            return
        if not 1 <= n <= self.max_n:
            messagebox.showerror(
                "Ошибка", f"Число участков должно быть от 1 до {self.max_n}.", parent=self
            )
            return
        self.result = {"n": n, "mode": self.v_mode.get()}
        self.destroy()


class PartitionApp(ImageApp):
    """Приложение первой лабораторной с функцией разбиения изображения."""

    def __init__(self):
        super().__init__()
        self.title("Цифровая обработка изображений")
        self.pixel_info.config(width=46)

        menubar = self.nametowidget(self.cget("menu"))
        part_menu = tk.Menu(menubar, tearoff=False)
        part_menu.add_command(
            label="Разбить на участки...", accelerator="Ctrl+R", command=self.partition
        )
        part_menu.add_separator()
        self.view_var = tk.StringVar(value="original")
        part_menu.add_radiobutton(
            label="Оригинал с границами", variable=self.view_var, value="original",
            command=lambda: self.show_partition_view("original")
        )
        part_menu.add_radiobutton(
            label="Цветная карта участков", variable=self.view_var, value="colored",
            command=lambda: self.show_partition_view("colored")
        )
        part_menu.add_separator()
        part_menu.add_command(label="Сохранить разметку...", command=self.save_labels)
        menubar.add_cascade(label="Разбиение", menu=part_menu)
        self.bind("<Control-r>", lambda event: self.partition())

        self._partition_original = None
        self._partition_labels = None
        self._partition_colored = None
        self._partition_bordered = None
        self._partition_seed = None

    @property
    def labels(self):
        return self._partition_labels

    def partition(self):
        """Разбить всё открытое изображение и по умолчанию показать оригинал с границами."""
        if self.image is None:
            messagebox.showinfo("Разбиение", "Сначала откройте изображение.")
            return

        params = self._ask_partition_params()
        if params is None:
            return

        original = self.image.copy()
        h, w = original.height, original.width
        seed = int.from_bytes(os.urandom(4), "little")
        self.canvas.config(cursor="watch")
        self.update_idletasks()
        try:
            labels = random_partition(h, w, params["n"], params["mode"], seed)
        except (ValueError, PartitionError) as exc:
            messagebox.showerror("Разбиение", str(exc))
            return
        finally:
            self.canvas.config(cursor="cross")

        # Сохраняем обычное состояние ImageApp в истории отмены.
        self.push_undo("разбиение на участки")
        self._partition_original = original
        self._partition_labels = labels
        self._partition_colored = labels_to_image(labels, seed)
        self._partition_bordered = draw_borders(original, labels)
        self._partition_seed = seed
        self.view_var.set("original")

        kind = "прямоугольники" if params["mode"] == "rect" else "произвольные фигуры"
        self.path = f"разбиение ({kind}, n={params['n']})"
        self._display(self._partition_bordered)
        self._update_status()

    def _ask_partition_params(self):
        w, h = self.image.size
        dialog = PartitionDialog(self, w * h)
        self.wait_window(dialog)
        return dialog.result

    def show_partition_view(self, view):
        """Переключить вид, не меняя сохранённую карту участков."""
        if self._partition_labels is None:
            return
        self.view_var.set(view)
        image = self._partition_bordered if view == "original" else self._partition_colored
        self._display(image)
        self._update_status()

    def save_labels(self):
        """Сохранить карту участков в формате .npy или .txt."""
        labels = self._partition_labels
        if labels is None:
            messagebox.showinfo("Сохранение", "Нет разметки: сначала выполните разбиение.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить разметку",
            defaultextension=".npy",
            filetypes=[("Массив NumPy", "*.npy"), ("Текст", "*.txt")],
        )
        if not path:
            return
        try:
            if os.path.splitext(path)[1].lower() == ".txt":
                np.savetxt(path, labels, fmt="%d")
            else:
                np.save(path, labels)
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{exc}")

    def _show_pixel_info(self, wx, wy):
        # При просмотре цветной карты показываем RGB исходного изображения.
        displayed = self.image
        if self._partition_original is not None and self._partition_labels is not None:
            self.image = self._partition_original
            try:
                super()._show_pixel_info(wx, wy)
            finally:
                self.image = displayed
            if self.pixel_info.cget("text"):
                x, y = self._canvas_to_pixel(wx, wy)
                if 0 <= x < self._partition_labels.shape[1] and 0 <= y < self._partition_labels.shape[0]:
                    self.pixel_info.config(
                        text=self.pixel_info.cget("text") +
                        f"  участок {int(self._partition_labels[y, x])}"
                    )
            return
        super()._show_pixel_info(wx, wy)

    def _update_status(self):
        super()._update_status()
        if self._partition_labels is not None and self.image is not None:
            self.status.config(
                text=self.status.cget("text") +
                f"  |  участков: {int(self._partition_labels.max())}"
            )

    def open_image(self):
        # Новое изображение сбрасывает сведения о предыдущем разбиении.
        super().open_image()
        # Если пользователь отменил открытие, сохраняем прежнее состояние.
        if self.image is not None and (
            self._partition_original is not None
            and self.image is not self._partition_bordered
            and self.image is not self._partition_colored
        ):
            self._clear_partition_state()

    def set_image(self, img):
        # set_image вызывается базовым приложением при открытии нового файла.
        # При открытии/обычной замене изображения очищаем метаданные разбиения.
        is_partition_view = img is self._partition_bordered or img is self._partition_colored
        if not is_partition_view and hasattr(self, "_partition_labels"):
            self._clear_partition_state()
        super().set_image(img)

    def undo(self):
        super().undo()
        # Если отмена вернула обычное изображение, старую разметку больше не показываем.
        if self.image is not self._partition_bordered and self.image is not self._partition_colored:
            self._clear_partition_state()
            self._update_status()

    def _clear_partition_state(self):
        self._partition_original = None
        self._partition_labels = None
        self._partition_colored = None
        self._partition_bordered = None
        self._partition_seed = None
        if hasattr(self, "view_var"):
            self.view_var.set("original")


if __name__ == "__main__":
    PartitionApp().mainloop()
