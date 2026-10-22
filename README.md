# UTS Pengolahan Citra Digital

Proyek ini berisi tiga program Python untuk pengolahan citra: menghitung selisih hasil filter Laplacian of Gaussian (LoG), segmentasi karakter, dan deteksi serta penghitungan koin berdasarkan ukuran.

## Struktur folder

```text
.
├── soal no 1/
│   ├── citra.png
│   ├── im1.png
│   ├── im2.png
│   ├── hasil.png
│   └── no1.py
├── soal no 2/
│   └── No2.py
└── soal no 3/
    ├── gambar_koin.png
    ├── hasil_koin.png
    └── no3.py
```

File gambar hasil yang sudah ada merupakan contoh hasil pengolahan. Menjalankan program dapat membuat atau menimpa file keluaran.

## Cara menjalankan proyek

1. Install Python 3 untuk Windows dari [python.org](https://www.python.org/downloads/). Saat instalasi, centang **Add python.exe to PATH**.
2. Buka folder proyek ini di VS Code.
3. Buka terminal VS Code dengan **Terminal > New Terminal**. Pastikan lokasi terminal berada di folder utama proyek (folder yang berisi `README.md` dan folder `soal no 1`, `soal no 2`, serta `soal no 3`).
4. Periksa bahwa Python tersedia:

   ```powershell
   python --version
   ```

   Jika perintah tidak dikenali, instal atau perbaiki instalasi Python dengan opsi **Add python.exe to PATH**, lalu tutup dan buka kembali VS Code.
5. Pasang pustaka yang diperlukan:

   ```powershell
   python -m pip install opencv-python numpy scipy scikit-image pillow
   ```

6. Jalankan program yang diinginkan menggunakan perintah pada bagian soal di bawah.

Contoh, untuk menjalankan program soal 2 dari folder utama proyek:

```powershell
python ".\soal no 2\No2.py"
```

Atau, jika terminal sudah berada di dalam folder `soal no 2`:

```powershell
python .\No2.py
```

Jalankan perintah `cd` berikut untuk berpindah dari folder utama ke folder soal 2:

```powershell
cd ".\soal no 2"
```

Setelah masuk folder tersebut, gunakan `cd ..` untuk kembali ke folder utama.

## Soal 1: Selisih filter LoG

Program membaca citra masukan dalam mode grayscale, menerapkan filter LoG dengan sigma `0.1` dan `0.2`, lalu menyimpan kedua respons dan citra selisihnya. Respons dinormalisasi agar dapat disimpan sebagai gambar 8-bit. Program juga mencetak statistik ke terminal.

```powershell
cd ".\soal no 1"
python .\no1.py citra.png
```

Keluaran bawaan tersimpan di folder `soal no 1` yang sama dengan program, sekalipun program dijalankan dari folder lain:

- `im1.png`: respons LoG sigma `0.1`
- `im2.png`: respons LoG sigma `0.2`
- `hasil_selisih_lg.png`: nilai absolut selisih kedua respons

Contoh menentukan nama file keluaran:

```powershell
python .\no1.py citra.png --im1 .\im1.png --im2 .\im2.png --output .\hasil.png
```

Opsi `--bandingkan` juga menghasilkan respons pembanding sigma `1` dan `2`:

```powershell
python .\no1.py citra.png --bandingkan
```

Jika terminal masih berada di folder utama proyek, jalankan dengan path folder:

```powershell
python ".\soal no 1\no1.py" ".\soal no 1\citra.png"
```

## Soal 2: Segmentasi karakter

Program ini membuka antarmuka grafis (GUI) untuk membuat gambar dari teks atau membuka gambar yang sudah ada, memisahkan karakter, dan menyimpan hasilnya.

Dari terminal di folder utama proyek, jalankan:

```powershell
python ".\soal no 2\No2.py"
```

Di jendela program, ketik teks lalu klik **Buat Gambar Teks**, atau klik **Buka Gambar...** untuk memilih gambar. Klik **Pecah Karakter** untuk menjalankan segmentasi. Untuk menyimpan hasil, klik **Simpan Tiap Karakter...** lalu pilih folder tujuan; program membuat subfolder `hasil_karakter` di dalamnya.

## Soal 3: Deteksi dan penghitungan koin

Program melakukan segmentasi koin dari latar, mengisi lubang pada objek, mengelompokkan objek berdasarkan diameter ekuivalen, lalu mencetak jumlah koin tiap kelompok. Keluaran gambar menandai koin dengan lingkaran dan nomor warna serta menambahkan legenda.

```powershell
cd ".\soal no 3"
python .\no3.py gambar_koin.png
```

Gambar hasil bawaan, `hasil_koin.png`, disimpan di folder `soal no 3` yang sama dengan program, sekalipun program dijalankan dari folder lain. Toleransi pengelompokan diameter bawaan adalah `0.08` (8%), dan area minimum objek adalah 50 piksel.

```powershell
python .\no3.py gambar_koin.png --toleransi 0.1 --min-area 50 --output .\hasil_koin.png
```

Jika terminal masih berada di folder utama proyek, jalankan:

```powershell
python ".\soal no 3\no3.py" ".\soal no 3\gambar_koin.png"
```

## Bantuan opsi

Untuk melihat seluruh opsi yang tersedia:

```powershell
python ".\soal no 1\no1.py" --help
python ".\soal no 3\no3.py" --help
```
