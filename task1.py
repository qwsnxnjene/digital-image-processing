import math
import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk

FILE_TYPES = [
    ("Изображения", "*.bmp *.png *.jpg *.jpeg"),
    ("BMP", "*.bmp"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
]

SAVE_FORMATS = {".bmp": "BMP", ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}
JPEG_QUALITY = 95
UNDO_LIMIT = 5

def image_to_arrays(img):
    img = img.convert("RGB")

    width, height = img.size

    r = np.empty((height, width), dtype=np.uint8)
    g = np.empty((height, width), dtype=np.uint8)
    b = np.empty((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            pixel = img.getpixel((x, y))

            r[y, x] = pixel[0]
            g[y, x] = pixel[1]
            b[y, x] = pixel[2]

    return r, g, b


def arrays_to_image(r, g, b):
    for name, ch in (("R", r), ("G", g), ("B", b)):
        if ch.dtype != np.uint8 or ch.ndim != 2:
            raise TypeError(f"Компонента {name} должна быть 2D-массивом uint8")
    if not (r.shape == g.shape == b.shape):
        raise ValueError("Компоненты R, G, B должны иметь одинаковую форму")
    return Image.fromarray(np.stack([r, g, b], axis=2), "RGB")


@dataclass
class Roi:
    x0: int
    y0: int
    mask: np.ndarray
    kind: str = "rect"                  # "rect" или "polygon"
    points: tuple = ()                  # вершины многоугольника (z1, z2)

    @staticmethod
    def rect(x0, y0, width, height):
        return Roi(x0, y0, np.ones((height, width), dtype=bool), "rect")

    @staticmethod
    def polygon(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0 = min(xs), min(ys)
        w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
        canvas = Image.new("L", (w, h), 0)
        ImageDraw.Draw(canvas).polygon([(x - x0, y - y0) for x, y in points],
                                       fill=255, outline=255)
        return Roi(x0, y0, np.asarray(canvas) > 0, "polygon", tuple(points))

    @property
    def width(self):
        return self.mask.shape[1]

    @property
    def height(self):
        return self.mask.shape[0]

    @property
    def x1(self):                       # правая граница (не включая)
        return self.x0 + self.width

    @property
    def y1(self):                       # нижняя граница (не включая)
        return self.y0 + self.height

    def crop(self, arr):
        return arr[self.y0:self.y1, self.x0:self.x1]

    def values(self, arr):
        return self.crop(arr)[self.mask]

    def full_mask(self, shape):
        m = np.zeros(shape[:2], dtype=bool)
        m[self.y0:self.y1, self.x0:self.x1] = self.mask
        return m


class ImageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Цифровая обработка изображений")
        self.geometry("1000x650")

        self.image = None   # текущее изображение: PIL.Image в режиме RGB (24 бит/пиксель)
        self.photo = None
        self.path = None    # путь к открытому файлу
        self.roi = None     # зона интереса
        self.roi_mode = tk.StringVar(value="rect")   # "rect" или "polygon"
        self._drag_start = None
        self._drag_end = None
        self._poly = []
        self._undo = []

        self._build_menu()
        self._build_canvas()
        self._build_statusbar()
        self._bind_scrolling()

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Открыть...", accelerator="Ctrl+O",
                              command=self.open_image)
        file_menu.add_command(label="Сохранить как...", accelerator="Ctrl+S",
                              command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.destroy)
        menubar.add_cascade(label="Файл", menu=file_menu)

        self.edit_menu = tk.Menu(menubar, tearoff=False)
        self.edit_menu.add_command(label="Отменить", accelerator="Ctrl+Z",
                                   command=self.undo, state="disabled")
        menubar.add_cascade(label="Правка", menu=self.edit_menu)

        image_menu = tk.Menu(menubar, tearoff=False)
        image_menu.add_command(label="Проверка: RGB -> 3 массива -> RGB",
                               command=self.check_roundtrip)
        menubar.add_cascade(label="Изображение", menu=image_menu)

        roi_menu = tk.Menu(menubar, tearoff=False)
        roi_menu.add_radiobutton(label="Прямоугольная (протянуть мышью)",
                                 variable=self.roi_mode, value="rect",
                                 command=self._cancel_selection)
        roi_menu.add_radiobutton(label="Многоугольник (клики по вершинам)",
                                 variable=self.roi_mode, value="polygon",
                                 command=self._cancel_selection)
        roi_menu.add_separator()
        roi_menu.add_command(label="Снять выделение", accelerator="Esc",
                             command=self.clear_roi)
        menubar.add_cascade(label="Зона интереса", menu=roi_menu)
        self.config(menu=menubar)
        self.bind("<Control-o>", lambda e: self.open_image())
        self.bind("<Control-s>", lambda e: self.save_image())
        self.bind("<Escape>", lambda e: self.clear_roi())
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Return>", lambda e: self._poly_finish())

    def _build_canvas(self):
        frame = tk.Frame(self)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(frame, bg="#606060", highlightthickness=0,
                                xscrollincrement=20, yscrollincrement=20,
                                cursor="cross")               # курсор в форме креста
        vbar = tk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        hbar = tk.Scrollbar(frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")

        self.canvas.bind("<Motion>", self._on_motion)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<Double-Button-1>", self._on_double)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Leave>", lambda e: self.pixel_info.config(text=""))

    def _build_statusbar(self):
        bar = tk.Frame(self, relief="sunken", bd=1)
        bar.pack(fill="x", side="bottom")

        self.pixel_info = tk.Label(bar, text="", anchor="w", padx=6, width=34,
                                   font="TkFixedFont")
        self.pixel_info.pack(side="right")

        self.status = tk.Label(bar, text="Файл не открыт", anchor="w", padx=6)
        self.status.pack(side="left", fill="x", expand=True)

    def _bind_scrolling(self):
        self.bind_all("<MouseWheel>", self._on_wheel)

    def _on_wheel(self, event):
        if event.num == 4:
            step = -1
        elif event.num == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1
        if event.state & 0x0001:
            self.canvas.xview_scroll(step, "units")
        else:
            self.canvas.yview_scroll(step, "units")
        self.after_idle(self._refresh_pixel_info)

    def _on_motion(self, event):
        self._show_pixel_info(event.x, event.y)
        if self._poly:
            self._poly_redraw(event.x, event.y)

    def _refresh_pixel_info(self):
        px = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        py = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        if self._drag_start is not None:
            self._update_drag(px, py)
        if self._poly:
            self._poly_redraw(px, py)
        if 0 <= px < self.canvas.winfo_width() and 0 <= py < self.canvas.winfo_height():
            self._show_pixel_info(px, py)

    def _canvas_to_pixel(self, wx, wy, clamp=False):
        z1 = math.floor(self.canvas.canvasx(wx))
        z2 = math.floor(self.canvas.canvasy(wy))
        if clamp:
            z1 = min(max(z1, 0), self.image.width - 1)
            z2 = min(max(z2, 0), self.image.height - 1)
        return z1, z2

    def _show_pixel_info(self, wx, wy):
        if self.image is None:
            return
        z1, z2 = self._canvas_to_pixel(wx, wy)
        if 0 <= z1 < self.image.width and 0 <= z2 < self.image.height:
            r, g, b = self.image.getpixel((z1, z2))
            self.pixel_info.config(text=f"({z1}, {z2})  R={r} G={g} B={b}")
        else:
            self.pixel_info.config(text="")

    def _on_press(self, event):
        if self.image is None:
            return
        if self.roi_mode.get() == "rect":
            self._rect_press(event)
        else:
            self._poly_click(event)

    def _on_double(self, event):
        if self.image is None:
            return
        if self.roi_mode.get() == "rect":
            self._rect_press(event)
        else:
            self._poly_finish()

    def _on_drag(self, event):
        self._show_pixel_info(event.x, event.y)
        if self.roi_mode.get() == "rect":
            self._update_drag(event.x, event.y)
        elif self._poly:
            self._poly_redraw(event.x, event.y)

    def _on_release(self, event):
        if self.roi_mode.get() == "rect":
            self._rect_release(event)

    def _rect_press(self, event):
        self._drag_start = self._canvas_to_pixel(event.x, event.y, clamp=True)
        self._drag_end = self._drag_start

    def _update_drag(self, wx, wy):
        if self._drag_start is None:
            return
        self._drag_end = self._canvas_to_pixel(wx, wy, clamp=True)
        x0, y0, x1, y1 = self._drag_rect()
        self.canvas.delete("roi")
        self._draw_outline("rect", x0, y0, x1, y1)

    def _drag_rect(self):
        (a1, a2), (b1, b2) = self._drag_start, self._drag_end
        return min(a1, b1), min(a2, b2), max(a1, b1) + 1, max(a2, b2) + 1

    def _rect_release(self, event):
        if self._drag_start is None:
            return
        self._update_drag(event.x, event.y)
        start, end = self._drag_start, self._drag_end
        x0, y0, x1, y1 = self._drag_rect()
        self._drag_start = None
        if start == end:                    # простой клик без протягивания
            self._apply_roi(None, "снятие выделения")
        else:
            self._apply_roi(Roi.rect(x0, y0, x1 - x0, y1 - y0),
                            "выделение прямоугольной зоны")

    def _poly_click(self, event):
        vertex = self._canvas_to_pixel(event.x, event.y, clamp=True)
        if not self._poly:
            self.canvas.delete("roi")
        if not self._poly or vertex != self._poly[-1]:
            self._poly.append(vertex)
        self._poly_redraw(event.x, event.y)
        self._update_status()

    def _poly_redraw(self, wx, wy):
        self.canvas.delete("roi")
        path = [(z1 + 0.5, z2 + 0.5) for z1, z2 in self._poly]
        cur = self._canvas_to_pixel(wx, wy, clamp=True)
        path.append((cur[0] + 0.5, cur[1] + 0.5))
        if len(path) >= 2:
            self._draw_outline("line", *[c for pt in path for c in pt])
        for cx, cy in path[:-1]:            # маркеры вершин
            self.canvas.create_rectangle(cx - 2, cy - 2, cx + 2, cy + 2,
                                         outline="black", fill="white", tags="roi")

    def _poly_finish(self):
        if not self._poly:
            return
        if len(self._poly) < 3:
            self.status.config(text="Многоугольнику нужно минимум 3 вершины")
            return
        points = self._poly
        self._poly = []
        self._apply_roi(Roi.polygon(points), "выделение зоны-многоугольника")

    def _draw_outline(self, shape, *coords):
        c = self.canvas
        for color, dash in (("white", None), ("black", (4, 4))):
            if shape == "rect":
                c.create_rectangle(*coords, outline=color, dash=dash, tags="roi")
            elif shape == "line":
                c.create_line(*coords, fill=color, dash=dash, tags="roi")
            else:
                c.create_polygon(*coords, outline=color, fill="", dash=dash, tags="roi")

    def clear_roi(self):
        if self._poly or self._drag_start is not None:
            self._cancel_selection()
        elif self.image is not None:
            self._apply_roi(None, "снятие выделения")

    def _cancel_selection(self):
        self._drag_start = None
        self._poly = []
        self._draw_roi_overlay()
        self._update_status()

    def _apply_roi(self, roi, description):
        if roi is not None or self.roi is not None:      # состояние действительно меняется
            self.push_undo(description)
            self.roi = roi
        self._draw_roi_overlay()
        self._update_status()

    def _draw_roi_overlay(self):
        self.canvas.delete("roi")
        r = self.roi
        if r is None:
            return
        if r.kind == "polygon":
            coords = [c for z1, z2 in r.points for c in (z1 + 0.5, z2 + 0.5)]
            self._draw_outline("polygon", *coords)
        else:
            self._draw_outline("rect", r.x0, r.y0, r.x1, r.y1)

    def push_undo(self, description):
        self._undo.append((description, self.path, self.image, self.roi))
        if len(self._undo) > UNDO_LIMIT:
            del self._undo[0]
        self._refresh_undo_menu()

    def undo(self):
        if self._poly or self._drag_start is not None:
            self._cancel_selection()
            return
        if not self._undo:
            return
        _, path, image, roi = self._undo.pop()
        if image is not self.image:
            self.path = path
            self._display(image)
        self.roi = roi
        self._draw_roi_overlay()
        self._update_status()
        self._refresh_undo_menu()

    def _refresh_undo_menu(self):
        if self._undo:
            label, state = f"Отменить: {self._undo[-1][0]}", "normal"
        else:
            label, state = "Отменить", "disabled"
        self.edit_menu.entryconfig(0, label=label, state=state)

    def open_image(self):
        path = filedialog.askopenfilename(title="Открыть изображение",
                                          filetypes=FILE_TYPES)
        if not path:
            return
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")             # приводим к 24-битному RGB
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{exc}")
            return
        self.push_undo("открытие изображения")
        self.path = path
        self.set_image(img)

    def save_image(self):
        if self.image is None:
            messagebox.showinfo("Сохранение", "Нет изображения для сохранения.")
            return
        initial = os.path.splitext(os.path.basename(self.path))[0] if self.path else "image"
        path = filedialog.asksaveasfilename(
            title="Сохранить изображение",
            initialfile=initial,
            defaultextension=".png",
            filetypes=FILE_TYPES[1:],
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in SAVE_FORMATS:
            messagebox.showerror("Ошибка", "Поддерживаются расширения: "
                                 ".bmp, .png, .jpg, .jpeg")
            return
        try:
            if SAVE_FORMATS[ext] == "JPEG":
                self.image.save(path, "JPEG", quality=JPEG_QUALITY)
            else:
                self.image.save(path, SAVE_FORMATS[ext])
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{exc}")
            return
        self.path = path
        self._update_status()

    def check_roundtrip(self):
        if self.image is None:
            messagebox.showinfo("Проверка", "Сначала откройте изображение.")
            return
        r, g, b = image_to_arrays(self.image)
        back = arrays_to_image(r, g, b)
        same = np.array_equal(np.asarray(self.image), np.asarray(back))
        h, w = r.shape
        messagebox.showinfo(
            "Проверка RGB <-> 3 массива",
            f"Массивы R, G, B: форма ({h}, {w}), тип {r.dtype}\n"
            f"Диапазон R: {r.min()}..{r.max()}, G: {g.min()}..{g.max()}, "
            f"B: {b.min()}..{b.max()}\n"
            f"Обратное преобразование даёт то же изображение: "
            f"{'да' if same else 'НЕТ'}")

    def set_image(self, img):
        self.roi = None
        self._poly = []
        self._drag_start = None
        self._display(img)
        self._draw_roi_overlay()
        self._update_status()

    def _display(self, img):
        self.image = img
        self.canvas.delete("image")
        if img is None:
            self.photo = None
            self.canvas.configure(scrollregion=(0, 0, 0, 0))
        else:
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo, tags="image")
            self.canvas.tag_lower("image")
            self.canvas.configure(scrollregion=(0, 0, img.width, img.height))
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.pixel_info.config(text="")

    def _update_status(self):
        if self.image is None:
            self.status.config(text="Файл не открыт")
            self.title("Цифровая обработка изображений")
            return
        w, h = self.image.size
        if self._poly:
            n = len(self._poly)
            self.status.config(text=f"Многоугольник: вершин {n}  |  двойной клик "
                                    f"или Enter - замкнуть  |  Esc - отмена")
        else:
            text = f"{os.path.basename(self.path)}  |  {w} x {h}"
            if self.roi is not None:
                r = self.roi
                if r.kind == "polygon":
                    text += (f"  |  Зона: многоугольник, x={r.x0}, y={r.y0}, "
                             f"{r.width} x {r.height}, {int(r.mask.sum())} пикс.")
                else:
                    text += f"  |  Зона: x={r.x0}, y={r.y0}, {r.width} x {r.height}"
            self.status.config(text=text)
        self.title(f"Цифровая обработка изображений - {self.path}")


if __name__ == "__main__":
    ImageApp().mainloop()