import math
import os
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from skimage import filters, measure, morphology


# BAGIAN 1 : MEMBUAT GAMBAR TEKS DARI INPUT
def muat_font(ukuran):
    """Cari font TrueType yang tersedia; jika tidak ada pakai font bawaan."""
    for nama in ["arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf",
                 "DejaVuSans.ttf", "Helvetica.ttc"]:
        try:
            return ImageFont.truetype(nama, ukuran)
        except OSError:
            continue
    try:
        return ImageFont.load_default(ukuran)
    except TypeError:
        return ImageFont.load_default()


def gambar_karakter(ch, font, ukuran, sudut):
    """Render satu karakter pada layer persegi lalu putar sebesar 'sudut' derajat."""
    s = ukuran * 2
    im = Image.new("L", (s * 2, s * 2), 0)
    ImageDraw.Draw(im).text((s, s), ch, fill=255, font=font, anchor="mm")
    im = im.rotate(sudut, resample=Image.BICUBIC)
    return np.array(im)


def tempel(target, layer, cx, cy):
    """Tempel 'layer' ke 'target' dengan pusat (cx, cy) memakai operasi maksimum."""
    h, w = layer.shape
    x0, y0 = int(cx - w // 2), int(cy - h // 2)
    tx0, ty0 = max(x0, 0), max(y0, 0)
    tx1, ty1 = min(x0 + w, target.shape[1]), min(y0 + h, target.shape[0])
    if tx1 <= tx0 or ty1 <= ty0:
        return
    sub = layer[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0]
    target[ty0:ty1, tx0:tx1] = np.maximum(target[ty0:ty1, tx0:tx1], sub)


def buat_gambar_teks(teks, ukuran=28, acak=True):
    """Ubah teks input menjadi gambar grayscale (PIL 'L')."""
    kata_list = teks.split()
    if not kata_list:
        raise ValueError("Teks masih kosong.")

    font = muat_font(ukuran)
    jarak = int(ukuran * 1.9)      # jarak antar-karakter dalam satu kata
    m = ukuran * 2                 # margin dari tepi gambar
    rng = random.Random()

    if acak:
        W, H = 760, 520
    else:
        W = max(400, max(len(k) for k in kata_list) * jarak + 2 * m)
        H = max(200, 2 * m + len(kata_list) * int(ukuran * 2.2))

    kanvas = np.zeros((H, W), np.uint8)

    for idx, kata in enumerate(kata_list):
        n = len(kata)

        # ---- mode rapi: tiap kata satu baris, tanpa rotasi ----
        if not acak:
            y = m + idx * int(ukuran * 2.2)
            for i, ch in enumerate(kata):
                tempel(kanvas, gambar_karakter(ch, font, ukuran, 0), m + i * jarak, y)
            continue

        # ---- mode acak: kata ditempatkan miring & tidak boleh bertabrakan ----
        zona_terlarang = morphology.dilation(
            kanvas, morphology.disk(max(3, ukuran // 5))) > 0
        berhasil = False

        for _ in range(400):
            theta = rng.uniform(-55, 55)                  # arah baris (derajat)
            dx = math.cos(math.radians(theta))
            dy = -math.sin(math.radians(theta))

            # kecilkan jarak bila kata terlalu panjang untuk kanvas
            j = jarak
            if n > 1:
                batas_x = (W - 2 * m) / (abs(dx) * (n - 1)) if abs(dx) > 1e-6 else 1e9
                batas_y = (H - 2 * m) / (abs(dy) * (n - 1)) if abs(dy) > 1e-6 else 1e9
                j = min(jarak, batas_x, batas_y)
                if j < ukuran * 1.3:      # terlalu rapat -> huruf bisa menempel
                    continue
            L = (n - 1) * j

            x_lo, x_hi = (m, W - m - L * dx) if dx >= 0 else (m - L * dx, W - m)
            y_lo, y_hi = (m, H - m - L * dy) if dy >= 0 else (m - L * dy, H - m)
            if x_lo > x_hi or y_lo > y_hi:
                continue

            xs, ys = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
            temp = np.zeros_like(kanvas)
            for i, ch in enumerate(kata):
                sudut = theta + rng.uniform(-12, 12)
                tempel(temp, gambar_karakter(ch, font, ukuran, sudut),
                       xs + i * j * dx, ys + i * j * dy)

            if not np.any((temp > 100) & zona_terlarang):
                kanvas = np.maximum(kanvas, temp)
                berhasil = True
                break

        if not berhasil:
            raise ValueError("Teks terlalu panjang/banyak untuk kanvas. "
                             "Kecilkan ukuran font atau kurangi teks.")

    return Image.fromarray(kanvas, mode="L")


# BAGIAN 2 : SEGMENTASI KARAKTER DENGAN REGIONPROPS
def segmentasi(gambar_gray, luas_min=20, padding=3):
    """
    Pecah gambar menjadi karakter-karakter individual.
    Mengembalikan list dict: {'crop': PIL.Image, 'bbox': (r0,c0,r1,c1), 'area': int}
    """
    gray = np.array(gambar_gray)
    if gray.min() == gray.max():
        raise ValueError("Gambar kosong / hanya satu warna.")

    # 1. Thresholding Otsu -> citra biner
    biner = gray > filters.threshold_otsu(gray)
    if biner.mean() > 0.5:          # pastikan karakter = foreground (True)
        biner = ~biner

    # 2. Buang noise kecil
    biner = morphology.remove_small_objects(biner, min_size=luas_min)

    # 3. Labeling komponen terhubung: tiap karakter mendapat nomor unik
    label_img = measure.label(biner, connectivity=2)

    # 4. regionprops: properti tiap region (bbox, area, centroid, ...)
    regions = [r for r in measure.regionprops(label_img) if r.area >= luas_min]
    regions.sort(key=lambda r: (r.centroid[0], r.centroid[1]))  # atas->bawah, kiri->kanan

    # 5. Crop tiap karakter berdasarkan bounding box
    tinggi, lebar = gray.shape
    hasil = []
    for r in regions:
        r0, c0, r1, c1 = r.bbox                  # (min_row, min_col, max_row, max_col)
        r0, c0 = max(r0 - padding, 0), max(c0 - padding, 0)
        r1, c1 = min(r1 + padding, tinggi), min(c1 + padding, lebar)
        crop = Image.fromarray(gray[r0:r1, c0:c1], mode="L")
        hasil.append({"crop": crop, "bbox": (r0, c0, r1, c1), "area": int(r.area)})
    return hasil


# BAGIAN 3 : GUI (Tkinter)
class Aplikasi(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Segmentasi Karakter dengan regionprops")
        self.geometry("1150x640")

        self.gambar = None      # PIL 'L' : gambar teks / gambar yang dibuka
        self.hasil = []         # hasil segmentasi

        self._bangun_ui()

    # ---------------------------------------------------------- UI
    def _bangun_ui(self):
        kiri = ttk.Frame(self, padding=10)
        kiri.pack(side="left", fill="y")
        kanan = ttk.Frame(self, padding=10)
        kanan.pack(side="right", fill="both", expand=True)

        ttk.Label(kiri, text="Teks input\n(pisahkan kata dengan spasi / Enter)").pack(anchor="w")
        self.txt = tk.Text(kiri, width=28, height=5, font=("Consolas", 12))
        self.txt.insert("1.0", "ZEUS TWINPEAK ACOSTA")
        self.txt.pack(pady=(2, 10))

        baris = ttk.Frame(kiri)
        baris.pack(anchor="w", pady=2)
        ttk.Label(baris, text="Ukuran font:").pack(side="left")
        self.var_ukuran = tk.IntVar(value=28)
        ttk.Spinbox(baris, from_=12, to=80, width=5, textvariable=self.var_ukuran).pack(side="left", padx=5)

        baris2 = ttk.Frame(kiri)
        baris2.pack(anchor="w", pady=2)
        ttk.Label(baris2, text="Luas minimum:").pack(side="left")
        self.var_luas = tk.IntVar(value=20)
        ttk.Spinbox(baris2, from_=1, to=500, width=5, textvariable=self.var_luas).pack(side="left", padx=5)

        self.var_acak = tk.BooleanVar(value=True)
        ttk.Checkbutton(kiri, text="Susun acak & miring (seperti contoh soal)",
                        variable=self.var_acak).pack(anchor="w", pady=6)

        ttk.Button(kiri, text="1. Buat Gambar Teks", command=self.buat_gambar).pack(fill="x", pady=3)
        ttk.Button(kiri, text="   atau Buka Gambar...", command=self.buka_gambar).pack(fill="x", pady=3)
        ttk.Button(kiri, text="2. Pecah Karakter", command=self.pecah).pack(fill="x", pady=3)
        ttk.Button(kiri, text="3. Simpan Tiap Karakter...", command=self.simpan).pack(fill="x", pady=3)

        self.status = tk.StringVar(value="Ketik teks lalu klik 'Buat Gambar Teks'.")
        ttk.Label(kiri, textvariable=self.status, wraplength=230, foreground="#0a5").pack(anchor="w", pady=15)

        # ---- tab di sisi kanan ----
        nb = ttk.Notebook(kanan)
        nb.pack(fill="both", expand=True)
        tab1, tab2 = ttk.Frame(nb), ttk.Frame(nb)
        nb.add(tab1, text="Gambar & Bounding Box")
        nb.add(tab2, text="Karakter Hasil Pecahan")
        self.nb = nb

        self.lbl_gambar = ttk.Label(tab1, anchor="center")
        self.lbl_gambar.pack(fill="both", expand=True)
        self.lbl_montase = ttk.Label(tab2, anchor="nw")
        self.lbl_montase.pack(fill="both", expand=True)

    # ------------------------------------------------------ helper tampil
    @staticmethod
    def _ke_photo(pil_img, maks=(800, 540)):
        im = pil_img.copy()
        im.thumbnail(maks)
        return ImageTk.PhotoImage(im)

    def _tampil_gambar(self, pil_img):
        foto = self._ke_photo(pil_img)
        self.lbl_gambar.configure(image=foto)
        self.lbl_gambar.image = foto

    # ------------------------------------------------------------ aksi
    def buat_gambar(self):
        teks = self.txt.get("1.0", "end").strip()
        try:
            self.gambar = buat_gambar_teks(teks, self.var_ukuran.get(), self.var_acak.get())
        except ValueError as e:
            messagebox.showwarning("Peringatan", str(e))
            return
        self.hasil = []
        self._tampil_gambar(self.gambar.convert("RGB"))
        self.lbl_montase.configure(image="")
        self.nb.select(0)
        self.status.set("Gambar teks dibuat. Klik 'Pecah Karakter'.")

    def buka_gambar(self):
        path = filedialog.askopenfilename(
            filetypes=[("Gambar", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Semua", "*.*")])
        if not path:
            return
        self.gambar = Image.open(path).convert("L")
        self.hasil = []
        self._tampil_gambar(self.gambar.convert("RGB"))
        self.lbl_montase.configure(image="")
        self.nb.select(0)
        self.status.set(f"Gambar dibuka: {os.path.basename(path)}")

    def pecah(self):
        if self.gambar is None:
            messagebox.showinfo("Info", "Buat atau buka gambar terlebih dahulu.")
            return
        try:
            self.hasil = segmentasi(self.gambar, luas_min=self.var_luas.get())
        except ValueError as e:
            messagebox.showwarning("Peringatan", str(e))
            return

        # --- gambar dengan bounding box ---
        rgb = self.gambar.convert("RGB")
        d = ImageDraw.Draw(rgb)
        for i, h in enumerate(self.hasil, start=1):
            r0, c0, r1, c1 = h["bbox"]
            d.rectangle([c0, r0, c1, r1], outline=(255, 0, 0), width=2)
            d.text((c0, max(r0 - 12, 0)), str(i), fill=(255, 255, 0))
        self._tampil_gambar(rgb)

        # --- montase karakter hasil pecahan ---
        self._tampil_montase()
        self.status.set(f"Ditemukan {len(self.hasil)} karakter. "
                        "Lihat tab 'Karakter Hasil Pecahan'.")

    def _tampil_montase(self):
        kolom, sel_w, sel_h = 8, 100, 115
        baris = max(1, math.ceil(len(self.hasil) / kolom))
        canvas = Image.new("RGB", (kolom * sel_w, baris * sel_h), (40, 40, 40))
        d = ImageDraw.Draw(canvas)
        for i, h in enumerate(self.hasil):
            thumb = h["crop"].copy()
            thumb.thumbnail((sel_w - 16, sel_h - 36))
            x = (i % kolom) * sel_w + (sel_w - thumb.width) // 2
            y = (i // kolom) * sel_h + 6
            canvas.paste(thumb.convert("RGB"), (x, y))
            d.text(((i % kolom) * sel_w + 8, (i // kolom) * sel_h + sel_h - 20),
                   f"#{i + 1}  {h['crop'].width}x{h['crop'].height}", fill=(255, 255, 255))
        foto = ImageTk.PhotoImage(canvas)
        self.lbl_montase.configure(image=foto)
        self.lbl_montase.image = foto

    def simpan(self):
        if not self.hasil:
            messagebox.showinfo("Info", "Belum ada hasil. Klik 'Pecah Karakter' dahulu.")
            return
        folder = filedialog.askdirectory(title="Pilih folder penyimpanan")
        if not folder:
            return
        folder = os.path.join(folder, "hasil_karakter")
        os.makedirs(folder, exist_ok=True)
        for i, h in enumerate(self.hasil, start=1):
            h["crop"].save(os.path.join(folder, f"karakter_{i:02d}.png"))
        self.gambar.save(os.path.join(folder, "_gambar_asli.png"))
        self.status.set(f"{len(self.hasil)} karakter tersimpan di:\n{folder}")
        messagebox.showinfo("Selesai", f"{len(self.hasil)} karakter disimpan di:\n{folder}")


if __name__ == "__main__":
    Aplikasi().mainloop()