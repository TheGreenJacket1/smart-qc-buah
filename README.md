# Smart QC Buah

Aplikasi kasir dan sortir kualitas buah berbasis Streamlit + YOLOv8: upload foto atau kamera realtime, deteksi buah segar/busuk, hitung harga, simpan riwayat di SQLite, dashboard, koreksi staf, dan ekspor laporan.

> Membuka `index.html` atau file HTML lain dengan klik dua kali **tidak akan berfungsi**. Aplikasi ini butuh server Python yang menjalankan model.

## Struktur

```
smart-qc-buah/
  app.py                     # seluruh aplikasi: logika, UI, kamera realtime
  models/best.pt             # bobot YOLOv8, 10 kelas buah
  tests/test_app.py          # tes logika dengan stub model
  .streamlit/config.toml     # batas upload 10 MB, tema, telemetri mati
  data/                      # dibuat otomatis: detections.db, recent/, review_queue/
  requirements.txt           # dependensi aplikasi
  requirements-dev.txt       # dependensi + pytest
  Dockerfile                 # image untuk deploy (port 7860)
```

## Menjalankan

Butuh Python 3.10–3.14.

```powershell
git clone https://github.com/TheGreenJacket1/smart-qc-buah.git
cd smart-qc-buah
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Buka <http://localhost:8501>. Hentikan dengan `Ctrl+C`.

Jika PowerShell menolak aktivasi venv:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Kalau `models/best.pt` belum ada, letakkan bobot di folder `models/`. Aplikasi tetap terbuka, tetapi deteksi menolak berjalan sampai model tersedia.

Akses dari HP di jaringan yang sama:

```powershell
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Lalu buka `http://IP-KOMPUTER:8501`.

## Fitur

- **Kasir** — upload banyak foto JPG/PNG (maks 10 MB per file), slider confidence dan IoU, filter jenis buah.
- **Deteksi** — kotak anotasi, status segar/busuk, daftar objek, harga, total, dan latensi.
- **Kamera realtime** — `streamlit-webrtc` dengan FPS dan latensi; frame live tidak dicatat ke database.
- **Keranjang dan struk** — tambah otomatis jika semua segar, diskon, uang diterima, kembalian, struk `.txt`.
- **Koreksi staf** — Rotten (Afkir), False Positive, False Negative; satu scan hanya boleh satu keputusan.
- **Dashboard** — KPI, objek per kelas, histogram confidence, scan 14 hari terakhir.
- **Riwayat** — sorting, filter status, paginasi, ekspor CSV/XLSX/JSON.
- **Log retur** — pencarian, tabel afkir/koreksi, ekspor CSV.

## Kamera realtime

- Pilih sumber `Kamera`, tekan **Start**, lalu izinkan akses kamera di browser.
- Kotak hijau = segar, merah = busuk. Pratinjau live **tidak** disimpan ke database.
- **Ambil dan catat** mengambil frame terakhir dan menyimpannya sebagai scan seperti upload.
- Browser hanya mengizinkan kamera di `localhost` atau HTTPS.
- Tanpa paket `streamlit-webrtc`, aplikasi otomatis memakai `st.camera_input` (foto tunggal, bukan video).

## Tes

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tes memakai stub model, jadi tidak perlu memuat PyTorch/YOLO.

## Konfigurasi

| Variabel | Default | Keterangan |
|---|---|---|
| `MODEL_PATH` | `models/best.pt` | Lokasi bobot YOLOv8 |
| `DB_PATH` | `data/detections.db` | Lokasi database SQLite |

Contoh:

```powershell
$env:MODEL_PATH="D:\model\best.pt"; python -m streamlit run app.py
```

## Kelas dan harga

10 kelas: `apple_fresh`, `apple_rotten`, `banana_fresh`, `banana_rotten`, `grape_fresh`, `grape_rotten`, `mango_fresh`, `mango_rotten`, `orange_fresh`, `orange_rotten`.

Harga per buah: apel 8.500, pisang 6.000, anggur 25.000, mangga 12.000, jeruk 7.500. Buah busuk dihargai 0 dan masuk daftar retur.

## Deploy

### Streamlit Community Cloud

1. Push repo ke GitHub.
2. Buka <https://share.streamlit.io>, pilih **New app**, arahkan ke repo dan `app.py`.
3. Tambahkan variabel `MODEL_PATH` bila bobot disimpan di tempat lain.

Filesystem Streamlit Cloud bersifat sementara; database dan gambar bisa hilang saat app restart/rebuild.

### Docker / Hugging Face Spaces

```powershell
docker build -t smart-qc-buah .
docker run -p 7860:7860 smart-qc-buah
```

Untuk Hugging Face Spaces pilih SDK **Docker**; `Dockerfile` sudah memakai port 7860 dan `$PORT`. Kamera realtime di HF Spaces memerlukan HTTPS (sudah disediakan) dan sebaiknya server TURN sendiri, karena STUN publik sering diblokir.

SQLite di platform gratis **tidak permanen**. Untuk produksi, ganti `db()` di `app.py` ke PostgreSQL/MySQL atau pasang volume persisten.

## Catatan

- Penjualan disimpan di tabel `sales` pada SQLite, bukan `localStorage`.
- Isi keranjang ada di `st.session_state`, jadi hilang jika tab di-refresh.
- Struk diunduh sebagai `.txt`; untuk mencetak gunakan Ctrl+P.
- Tempel gambar dari clipboard (Ctrl+V) belum didukung.
- Kamera dan upload berbagi satu instance model dengan lock, jadi inferensi bergantian, bukan paralel.

## Kontributor

Proyek dikerjakan bersama. Agar kontribusi tercatat di GitHub, commit harus memakai email yang terhubung ke akun GitHub masing-masing (cek **Settings → Emails**), lalu push ke `main` atau lewat pull request yang di-merge.

## 📌 6 Pertanyaan Wajib

### 1. Masalahnya apa?
* **Deskripsi Masalah:** Buah yang mengalami pembusukan, memar, atau berjamur kerap kali luput dari pengawasan staf dan tetap terpajang di rak display atau lolos saat penimbangan di kasir swalayan.
* **Pihak yang Dirugikan:**
  * **Konsumen:** Mendapatkan produk yang tidak higienis dan tidak layak konsumsi, yang berujung pada kekecewaan dan komplain.
  * **Manajemen Swalayan:** Mengalami kerugian reputasi, potensi penurunan penjualan, serta risiko penularan pembusukan ke buah sehat lainnya dalam satu wadah/keranjang.
* **Frekuensi:** Terjadi setiap hari, terutama pada jam-jam sibuk (*peak hours*) dan saat pergantian shift ketika staf tidak sempat menyortir produk secara manual satu per satu.

---

### 2. Apa yang dideteksi?
* **Objek & Kondisi:** Citra buah di atas meja sortir/timbangan swalayan. Sistem mendeteksi jenis buah beserta kondisi mutunya ke dalam 10 kategori:
  * Buah Segar (*Fresh*): `apple_fresh`, `banana_fresh`, `grape_fresh`, `mango_fresh`, `orange_fresh`.
  * Buah Busuk/Afkir (*Rotten*): `apple_rotten`, `banana_rotten`, `grape_rotten`, `mango_rotten`, `orange_rotten`.
* **Input Visual:** Tangkapan kamera (webcam/kamera pengawas meja timbang) yang menyorot objek buah secara langsung.

---

### 3. Setelah itu, lalu apa? (Tindakan Sistem)
* **Jika Semua Buah Terdeteksi Segar (`_fresh`):**
  * Sistem memberikan indikator visual hijau (*"Layak Jual"*).
  * Menghitung estimasi total harga dan mengaktifkan fitur pencetakan label barcode harga untuk ditempel pada kemasan.
* **Jika Terdeteksi Buah Rusak/Afkir (`_rotten`):**
  * Sistem memicu peringatan darurat (*visual alert* berwarna merah).
  * **Memblokir pencetakan barcode harga** secara otomatis agar produk cacat tidak bisa diproses ke pembayaran kasir.
  * Mencatat item yang rusak ke dalam database log limbah (`waste_log.csv`) secara otomatis lengkap dengan *timestamp*, tingkat keyakinan (*confidence*), dan status disposisi untuk mempermudah klaim retur ke pihak *supplier*.

---

### 4. Apa yang berubah? (Metrik Keberhasilan)
* **Kualitas Layanan:** Menekan angka keluhan (*customer complaint*) terkait buah busuk hingga mendekati 0%.
* **Efisiensi Operasional:** Mempercepat proses pemilahan mutu buah dari rata-rata 3–5 detik per buah menjadi di bawah 1 detik dengan bantuan asistensi Computer Vision.
* **Akuntabilitas Data:** Manajemen swalayan memiliki data rekap barang afkir (*waste/shrinkage log*) yang tercatat otomatis dan terstruktur tanpa perlu pencatatan formulir kertas manual.

---

### 5. Kalau sistem salah? (Penanganan Eror / Human-in-the-Loop)
* **Salah Deteksi / False Positive (Buah segar dikira busuk):**
  * *Penyebab:* Bintik manis alami kulit pisang atau bayangan/pantulan cahaya tajam yang menyerupai jamur.
  * *Penanganan:* Disediakan tombol **"⚠️ Override Manual (Sistem Salah)"**. Staf kasir/sortir dapat membatalkan status afkir setelah memeriksa fisik buah, sehingga barcode tetap dapat dicetak. Koreksi staf ini otomatis tercatat di log audit.
* **Terlewat / False Negative (Buah busuk dikira segar):**
  * *Penyebab:* Bagian busuk berada di sisi bawah yang tidak tertangkap sudut kamera.
  * *Penanganan:* Disediakan tombol **"🚫 Laporkan Kerusakan (Terlewat)"** agar staf dapat menarik produk secara manual dan memasukkannya ke daftar log barang rusak. Staf juga diinstruksikan untuk memutar posisi buah jika tingkat keyakinan (*confidence*) berada di rentang ambang batas bawah.

---

### 6. Data siapa yang direkam? (Privasi & Keamanan)
* **Objek Rekaman:** Kamera diposisikan mengarah tegak lurus ke nampan/meja timbangan (*downward-facing camera*), sehingga **hanya merekam objek buah**.
* **Privasi Pengunjung & Staf:** Sistem tidak merekam maupun memproses wajah staf atau pengunjung toko.
* **Penyimpanan Data:** Frame video diproses langsung di memori (*in-memory processing*). Data yang disimpan ke media penyimpanan permanen hanyalah metadata log operasional (`Timestamp`, `Jenis Buah`, `Status Deteksi`, `Confidence`, `Tindakan`) ke dalam file CSV untuk keperluan audit inventaris.
