"""Hitung jumlah koin untuk setiap ukuran menggunakan regionprops.

Pseudocode:
    1. Baca citra, haluskan (Gaussian blur).
    2. Hitung jarak warna tiap piksel ke warna latar, threshold Otsu
       -> citra biner (koin vs latar).
    3. Warna latar = median piksel di tepi citra (PNG transparan: pakai alpha).
    4. Isi lubang di dalam koin.
    5. Beri label tiap objek (connected components): label().
    6. Untuk tiap objek, ambil area dan diameter ekuivalen: regionprops(),
       lalu buang objek yang areanya terlalu kecil (noise).
    7. Urutkan diameter, pisahkan jadi kelompok ukuran bila selisih antar
       diameter berurutan melebihi toleransi.
    8. Hitung jumlah koin per kelompok, cetak tabel, simpan citra berlabel.

Contoh:
    python no2.py koin.png
    python no2.py koin.png --toleransi 0.1 --output hasil_koin.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def segmentasi(citra: np.ndarray) -> np.ndarray:
    """Hasilkan citra biner: koin = True, latar = False.

    Koin dipisahkan dari latar berdasarkan JARAK WARNA terhadap warna latar,
    bukan kecerahan, supaya koin yang punya banyak warna (tepi terang, isi
    gelap, simbol) tetap terbaca sebagai satu objek utuh.
    """
    # PNG transparan: alpha sudah menyatakan mana koin dan mana latar
    if citra.ndim == 3 and citra.shape[2] == 4 and citra[:, :, 3].min() < 255:
        return ndi.binary_fill_holes(citra[:, :, 3] > 127)

    bgr = citra[:, :, :3] if citra.ndim == 3 else cv2.cvtColor(citra, cv2.COLOR_GRAY2BGR)
    bgr = cv2.GaussianBlur(bgr, (5, 5), 0).astype(np.float32)

    # Warna latar = median piksel di keempat tepi citra
    tepi = np.concatenate([bgr[0, :], bgr[-1, :], bgr[:, 0], bgr[:, -1]])
    warna_latar = np.median(tepi, axis=0)

    jarak = np.linalg.norm(bgr - warna_latar, axis=2)
    jarak = np.clip(jarak, 0, 255).astype(np.uint8)
    biner = jarak > threshold_otsu(jarak)

    return ndi.binary_fill_holes(biner)


def kelompokkan(diameter: np.ndarray, toleransi: float) -> np.ndarray:
    """Beri nomor kelompok ukuran (0 = terkecil) berdasarkan diameter."""
    urut = np.argsort(diameter)
    kelompok = np.zeros(len(diameter), dtype=int)
    nomor = 0
    for i in range(1, len(urut)):
        if diameter[urut[i]] > diameter[urut[i - 1]] * (1 + toleransi):
            nomor += 1
        kelompok[urut[i]] = nomor
    return kelompok


WARNA = [(0, 0, 230), (220, 90, 0), (0, 150, 0), (190, 0, 190),
         (0, 140, 255), (190, 190, 0), (60, 60, 60), (0, 100, 160)]  # BGR


def gambar_hasil(citra, objek, kelompok, diameter, baris_legenda) -> np.ndarray:
    """Gambar koin dengan lingkaran berwarna per ukuran, plus panel legenda."""
    tinggi, lebar = citra.shape[:2]
    skala = max(0.6, tinggi / 900)
    tebal = max(1, int(round(skala * 2)))
    hasil = citra.copy()

    for o, k, d in zip(objek, kelompok, diameter):
        y, x = map(int, o.centroid)
        warna = WARNA[k % len(WARNA)]
        cv2.circle(hasil, (x, y), int(d / 2), warna, tebal, cv2.LINE_AA)
        teks = str(k + 1)
        (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * skala, tebal)
        cv2.putText(hasil, teks, (x - tw // 2, y + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * skala, warna, tebal, cv2.LINE_AA)

    # Panel legenda di sisi kanan
    lebar_panel = int(420 * skala)
    panel = np.full((tinggi, lebar_panel, 3), 255, np.uint8)
    y = int(50 * skala)
    cv2.putText(panel, f"Jumlah koin: {len(objek)}", (int(15 * skala), y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * skala, (0, 0, 0), tebal, cv2.LINE_AA)
    for k, teks in enumerate(baris_legenda):
        y += int(55 * skala)
        warna = WARNA[k % len(WARNA)]
        cv2.circle(panel, (int(30 * skala), y - int(8 * skala)), int(14 * skala),
                   warna, tebal, cv2.LINE_AA)
        cv2.putText(panel, str(k + 1), (int(24 * skala), y - int(1 * skala)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * skala, warna, tebal, cv2.LINE_AA)
        cv2.putText(panel, teks, (int(58 * skala), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * skala, (0, 0, 0), tebal, cv2.LINE_AA)
    return np.hstack([hasil, panel])


def main() -> None:
    folder_program = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Hitung koin per ukuran.")
    parser.add_argument("image", type=Path, help="Path citra masukan")
    parser.add_argument("-o", "--output", type=Path, default=folder_program / "hasil_koin.png",
                        help="Path citra berlabel (default: di folder program)")
    parser.add_argument("--toleransi", type=float, default=0.08,
                        help="Toleransi relatif diameter satu ukuran (default: 0.08)")
    parser.add_argument("--min-area", type=int, default=50,
                        help="Area minimum objek agar dihitung koin (default: 50)")
    args = parser.parse_args()

    citra_asli = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if citra_asli is None:
        parser.error(f"Citra tidak dapat dibaca: {args.image}")
    citra = cv2.imread(str(args.image), cv2.IMREAD_COLOR)  # untuk anotasi hasil

    biner = segmentasi(citra_asli)
    label_img = label(biner)
    objek = [o for o in regionprops(label_img) if o.area >= args.min_area]  # buang noise
    if not objek:
        parser.error("Tidak ada koin yang terdeteksi.")

    area = np.array([o.area for o in objek], dtype=float)
    diameter = np.array([o.equivalent_diameter_area for o in objek], dtype=float)
    kelompok = kelompokkan(diameter, args.toleransi)

    # Hasil di terminal
    print(f"Jumlah koin terdeteksi: {len(objek)}")
    print("Jumlah koin per kelompok diameter (piksel):")
    baris_legenda = []
    for k in range(kelompok.max() + 1):
        m = kelompok == k
        teks = f"sekitar {diameter[m].mean():.0f} px: {m.sum()} koin"
        print(f"  {teks}")
        baris_legenda.append(teks)

    # Hasil dalam bentuk gambar (koin diberi warna per ukuran + legenda)
    hasil = gambar_hasil(citra, objek, kelompok, diameter, baris_legenda)
    if not cv2.imwrite(str(args.output), hasil):
        parser.error(f"Citra hasil tidak dapat disimpan: {args.output}")
    print(f"Citra berlabel disimpan ke: {args.output}")


if __name__ == "__main__":
    main()