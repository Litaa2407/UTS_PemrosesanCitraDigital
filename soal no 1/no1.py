"""Hitung dan simpan selisih dua filter Laplacian of Gaussian (LoG).

Algoritma:
    a. Baca citra.
    b. Filter LoG sigma 0,1 -> im1.
    c. Filter LoG sigma 0,2 -> im2.
    d. Selisih im1 dan im2 -> citra hasil.

Contoh:
    python no1.py citra.png
    python no1.py citra.png --output hasil.png
    python no1.py citra.png --bandingkan   # tambahan: sigma 1 dan 2 sebagai pembanding
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace


def to_display_image(image: np.ndarray) -> np.ndarray:
    """Normalisasi magnitudo respons LoG ke 0-255 agar bisa disimpan dan dilihat."""
    magnitude = np.abs(image)
    maximum = float(np.max(magnitude))
    if maximum == 0:
        return np.zeros(magnitude.shape, dtype=np.uint8)
    return np.rint(magnitude / maximum * 255).astype(np.uint8)


def hitung_selisih(gray: np.ndarray, sigma1: float, sigma2: float):
    """Kembalikan (im1, im2, selisih) dalam nilai mentah (float)."""
    im1 = gaussian_laplace(gray, sigma=sigma1)
    im2 = gaussian_laplace(gray, sigma=sigma2)
    return im1, im2, np.abs(im1 - im2)


def cetak_statistik(gray, im1, im2, selisih, sigma1, sigma2) -> None:
    """Cetak rasio terhadap citra asli dan selisih maksimum (untuk laporan)."""
    mask = gray > 10  # hindari piksel terlalu gelap agar rasio stabil
    if mask.any():
        print(f"Rasio im1/asli (sigma {sigma1}): {np.mean(im1[mask] / gray[mask]):.2f}")
        print(f"Rasio im2/asli (sigma {sigma2}): {np.mean(im2[mask] / gray[mask]):.2f}")
    print(f"Selisih maksimum (mentah): {selisih.max():.4f}")


def main() -> None:
    folder_program = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Selisih hasil filter Laplacian of Gaussian sigma 0,1 dan 0,2."
    )
    parser.add_argument("image", type=Path, help="Path citra masukan")
    parser.add_argument("-o", "--output", type=Path,
                        default=folder_program / "hasil_selisih_lg.png",
                        help="Path citra hasil (default: di folder program)")
    parser.add_argument("--im1", type=Path, default=folder_program / "im1.png",
                        help="Path hasil LoG sigma 0,1 (default: di folder program)")
    parser.add_argument("--im2", type=Path, default=folder_program / "im2.png",
                        help="Path hasil LoG sigma 0,2 (default: di folder program)")
    parser.add_argument("--bandingkan", action="store_true",
                        help="Simpan juga hasil pembanding dengan sigma 1 dan 2")
    args = parser.parse_args()

    # a. Baca citra (grayscale)
    image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        parser.error(f"Citra tidak dapat dibaca: {args.image}")
    gray = image.astype(np.float32)

    # b, c, d. LoG sigma 0,1 -> im1; sigma 0,2 -> im2; selisih -> hasil
    im1, im2, selisih = hitung_selisih(gray, 0.1, 0.2)

    outputs = [
        (args.im1, to_display_image(im1)),
        (args.im2, to_display_image(im2)),
        (args.output, to_display_image(selisih)),
    ]

    print("=== Sigma 0,1 vs 0,2 (sesuai soal) ===")
    cetak_statistik(gray, im1, im2, selisih, 0.1, 0.2)

    # Opsional: pembanding sigma 1 vs 2 supaya efek LoG sebagai deteksi tepi terlihat
    if args.bandingkan:
        c1, c2, cd = hitung_selisih(gray, 1.0, 2.0)
        outputs += [
            (folder_program / "pembanding_im1_sigma1.png", to_display_image(c1)),
            (folder_program / "pembanding_im2_sigma2.png", to_display_image(c2)),
            (folder_program / "pembanding_selisih_sigma1_2.png", to_display_image(cd)),
        ]
        print("=== Pembanding sigma 1 vs 2 ===")
        cetak_statistik(gray, c1, c2, cd, 1.0, 2.0)

    for path, output_image in outputs:
        if not cv2.imwrite(str(path), output_image):
            parser.error(f"Citra hasil tidak dapat disimpan: {path}")
        print(f"Disimpan: {path}")


if __name__ == "__main__":
    main()