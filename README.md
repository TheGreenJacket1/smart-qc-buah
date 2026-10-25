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
