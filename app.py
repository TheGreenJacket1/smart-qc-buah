"""Smart QC Buah - aplikasi Streamlit untuk sortir kualitas buah.

Jalankan dari folder ini:
    streamlit run app.py

Satu file berisi tiga lapisan:
  1. logika (database, model YOLO, deteksi, ekspor) tanpa ketergantungan UI,
  2. kamera realtime (streamlit-webrtc),
  3. antarmuka Streamlit (Kasir, Daftar harga, Dashboard, Riwayat, Log retur).
Fungsi logika sengaja bebas dari st.* agar bisa diuji (lihat tests/test_app.py).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:  # kamera realtime bersifat opsional; tanpa paket ini dipakai st.camera_input
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    HAS_WEBRTC = True
except ImportError:  # pragma: no cover
    av = None
    webrtc_streamer = None
    VideoProcessorBase = object
    HAS_WEBRTC = False

# ------------------------------------------------------------------ konfigurasi
ROOT = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "models" / "best.pt"))
DB_PATH = os.getenv("DB_PATH", str(ROOT / "data" / "detections.db"))
RECENT_DIR = ROOT / "data" / "recent"          # buffer gambar terakhir
REVIEW_DIR = ROOT / "data" / "review_queue"    # kandidat data latih ulang

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_BATCH = 20
RECENT_KEEP = 200
LOW_CONF = 0.60  # deteksi di bawah ini masuk antrean review (active learning)
MAX_RESULT_CARDS = 6

HARGA_BUAH = {"apple": 8500, "banana": 6000, "grape": 25000,
              "mango": 12000, "orange": 7500}
NAMA = {"apple": "Apel", "banana": "Pisang", "grape": "Anggur",
        "mango": "Mangga", "orange": "Jeruk"}
FRUITS = list(HARGA_BUAH)
DEFAULT_PRICE = 10000

ACTION_BY_STATUS = {
    "Rotten (Afkir)": "Dibuang / Retur Supplier",
    "False Positive (Koreksi Staf)": "Override: Tetap Dijual",
    "False Negative (Koreksi Staf)": "Ditarik Manual",
}
STATUS_TXT = {"fresh": "Semua segar, layak jual", "rotten": "Ada buah busuk",
              "none": "Tidak ada buah terdeteksi"}
STATUS_COLOR = {"fresh": "green", "rotten": "red", "none": "orange"}
MODE_CART, MODE_CHECK = "Tambah ke keranjang", "Cek harga saja"
PAGES = ["Kasir", "Daftar harga", "Dashboard", "Riwayat", "Log retur"]


class AppError(Exception):
    """Galat yang pesannya aman ditampilkan ke pengguna."""


def nama(fruit: str) -> str:
    return NAMA.get(fruit, fruit)


def rp(n) -> str:
    return "Rp " + f"{int(n):,}".replace(",", ".")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------- database
def db() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with closing(db()) as con, con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS scans(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
            source TEXT, filename TEXT, n_objects INTEGER NOT NULL,
            status TEXT NOT NULL, latency_ms REAL);
        CREATE TABLE IF NOT EXISTS detections(
            id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER NOT NULL,
            ts TEXT NOT NULL, label TEXT NOT NULL, fruit TEXT NOT NULL,
            condition TEXT NOT NULL, confidence REAL NOT NULL,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL);
        CREATE TABLE IF NOT EXISTS waste_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER,
            ts TEXT NOT NULL, fruit TEXT NOT NULL, status TEXT NOT NULL,
            confidence REAL NOT NULL, action TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sales(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
            total INTEGER NOT NULL, qty INTEGER NOT NULL,
            discount_pct INTEGER NOT NULL, cash INTEGER NOT NULL, items TEXT NOT NULL);
        """)


@st.cache_resource
def ensure_db() -> bool:
    init_db()
    return True


# --------------------------------------------------------------------- model
class ModelStore:
    """Memuat YOLO sekali dan menserialkan inferensi (upload dan kamera berbagi model)."""

    def __init__(self, path: str):
        self.path = path
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is None:
            if not Path(self.path).exists():
                raise AppError(f"Model tidak ditemukan di {self.path}")
            try:
                from ultralytics import YOLO  # import lambat, jadi ditunda
            except ImportError:
                raise AppError("Paket ultralytics belum terpasang: pip install ultralytics")
            self._model = YOLO(self.path)
        return self._model

    def names(self) -> dict:
        with self._lock:
            return dict(self._load().names)

    def predict(self, img, conf, iou, class_ids, annotate):
        with self._lock:
            r = self._load().predict(img, conf=conf, iou=iou, classes=class_ids,
                                     imgsz=640, verbose=False)[0]
            dets = [(int(b.cls[0]), float(b.conf[0]), [float(v) for v in b.xyxy[0].tolist()])
                    for b in r.boxes]
            plotted = r.plot() if annotate else None
        return dets, plotted


@st.cache_resource
def model_store() -> ModelStore:
    return ModelStore(MODEL_PATH)


def get_names() -> dict:
    return model_store().names()


def predict_raw(img, conf, iou, class_ids, annotate=True):
    """Satu-satunya fungsi yang menyentuh YOLO (mudah di-mock saat tes)."""
    return model_store().predict(img, conf, iou, class_ids, annotate)


def split_label(label: str):
    fruit, _, cond = label.partition("_")
    return fruit, (cond or "unknown")


def normalize_classes(classes):
    """Terima None, string 'a,b' atau daftar nama; kembalikan list bersih atau None."""
    if classes is None:
        return None
    if isinstance(classes, str):
        items = classes.split(",")
    elif isinstance(classes, (list, tuple, set)):
        items = list(classes)
        if any(not isinstance(c, str) for c in items):
            raise AppError("classes harus berisi nama kelas berupa teks")
    else:
        raise AppError("classes harus berupa teks atau daftar nama kelas")
    return [c.strip() for c in items if c.strip()] or None


def resolve_classes(names: dict, wanted):
    wanted = normalize_classes(wanted)
    if not wanted:
        return None
    wanted_set = set(wanted)
    ids = [i for i, n in names.items() if n in wanted_set or n.split("_")[0] in wanted_set]
    if not ids:
        raise AppError(f"Kelas tidak dikenal. Pilihan: {sorted(names.values())}")
    return ids


def detect_image(img, conf, iou, classes, annotate=True) -> dict:
    names = get_names()
    class_ids = resolve_classes(names, classes)
    t0 = time.perf_counter()
    dets, annotated = predict_raw(img, conf, iou, class_ids, annotate)
    latency = (time.perf_counter() - t0) * 1000

    objects = []
    for cid, c, box in dets:
        label = names[cid]
        fruit, cond = split_label(label)
        objects.append({
            "label": label, "fruit": fruit, "condition": cond,
            "confidence": round(c, 4), "bbox": [round(v, 1) for v in box],
            "price": HARGA_BUAH.get(fruit, DEFAULT_PRICE) if cond != "rotten" else 0,
        })
    if not objects:
        status = "none"
    elif any(o["condition"] == "rotten" for o in objects):
        status = "rotten"
    else:
        status = "fresh"

    out = {"status": status, "n_objects": len(objects), "objects": objects,
           "total_price": sum(o["price"] for o in objects) if status == "fresh" else 0,
           "latency_ms": round(latency, 1),
           "settings": {"conf": conf, "iou": iou, "classes": classes}}
    if annotated is not None:  # st.image butuh RGB, YOLO menghasilkan BGR
        out["annotated_rgb"] = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return out


# --------------------------------------------------------------- penyimpanan
def store_image(scan_id: int, img, result: dict) -> None:
    """Simpan gambar terakhir; salin ke review_queue jika model ragu."""
    try:
        RECENT_DIR.mkdir(parents=True, exist_ok=True)
        path = RECENT_DIR / f"{scan_id}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        files = sorted(RECENT_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        for old in files[:-RECENT_KEEP]:
            old.unlink(missing_ok=True)
        confs = [o["confidence"] for o in result["objects"]]
        if not confs or min(confs) < LOW_CONF:
            REVIEW_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, REVIEW_DIR / f"{scan_id}_lowconf.jpg")
    except OSError:
        pass  # kegagalan simpan gambar tidak boleh menggagalkan deteksi


def save_scan(source: str, filename: str, result: dict, img=None) -> int:
    ts = now()
    with closing(db()) as con, con:
        cur = con.execute(
            "INSERT INTO scans(ts,source,filename,n_objects,status,latency_ms) VALUES(?,?,?,?,?,?)",
            (ts, source, filename, result["n_objects"], result["status"], result["latency_ms"]))
        sid = cur.lastrowid
        con.executemany(
            "INSERT INTO detections(scan_id,ts,label,fruit,condition,confidence,x1,y1,x2,y2) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            [(sid, ts, o["label"], o["fruit"], o["condition"], o["confidence"], *o["bbox"])
             for o in result["objects"]])
    if img is not None:
        store_image(sid, img, result)
    return sid


def decode_image(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise AppError("File bukan gambar yang valid (gunakan JPG atau PNG)")
    return img


def run_detection(data: bytes, filename: str, source: str, conf: float, iou: float,
                  classes=None, save: bool = True) -> dict:
    """Validasi, deteksi, dan simpan satu gambar (padanan POST /api/v1/detect)."""
    if not data:
        raise AppError("File kosong")
    if len(data) > MAX_IMAGE_BYTES:
        raise AppError("Ukuran gambar maksimal 10 MB")
    img = decode_image(data)
    result = detect_image(img, conf, iou, classes, True)
    result["filename"] = filename
    if save:
        result["scan_id"] = save_scan(source, filename, result, img)
    return result


# ---------------------------------------------- riwayat, statistik, ekspor
SCAN_SORT = {"timestamp": "ts", "n_objects": "n_objects", "latency": "latency_ms", "status": "status"}
DET_SORT = {"timestamp": "d.ts", "confidence": "d.confidence", "label": "d.label"}


def check_order(order: str) -> str:
    if str(order).lower() not in ("asc", "desc"):
        raise AppError("order harus asc atau desc")
    return order.upper()


def history(sort_by="timestamp", order="desc", status: Optional[str] = None,
            limit: int = 50, offset: int = 0) -> dict:
    if sort_by not in SCAN_SORT:
        raise AppError(f"sort_by harus salah satu dari {list(SCAN_SORT)}")
    where, params = ("WHERE status=?", [status]) if status else ("", [])
    with closing(db()) as con:
        rows = con.execute(
            f"SELECT * FROM scans {where} ORDER BY {SCAN_SORT[sort_by]} {check_order(order)}, id DESC "
            "LIMIT ? OFFSET ?", params + [limit, offset]).fetchall()
        total = con.execute(f"SELECT COUNT(*) FROM scans {where}", params).fetchone()[0]
    return {"total": total, "items": [dict(r) for r in rows]}


def _det_filter(label=None, min_conf=0.0) -> tuple[list[str], list]:
    cond, params = ["d.confidence>=?"], [min_conf]
    if label:
        cond.append("(d.label=? OR d.fruit=?)")
        params += [label, label]
    return cond, params


def det_query(label=None, min_conf=0.0, sort_by="timestamp", order="desc", limit=None, offset=0):
    if sort_by not in DET_SORT:
        raise AppError(f"sort_by harus salah satu dari {list(DET_SORT)}")
    cond, params = _det_filter(label, min_conf)
    sql = ("SELECT d.id, d.scan_id, d.ts, s.filename, d.label, d.fruit, d.condition, d.confidence, "
           "d.x1, d.y1, d.x2, d.y2 FROM detections d JOIN scans s ON s.id=d.scan_id "
           f"WHERE {' AND '.join(cond)} ORDER BY {DET_SORT[sort_by]} {check_order(order)}, d.id DESC")
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params += [limit, offset]
    with closing(db()) as con:
        return [dict(r) for r in con.execute(sql, params).fetchall()]


def det_count(label=None, min_conf=0.0) -> int:
    """Jumlah baris ekspor tanpa memuat seluruh data ke memori."""
    cond, params = _det_filter(label, min_conf)
    with closing(db()) as con:
        return con.execute(
            "SELECT COUNT(*) FROM detections d JOIN scans s ON s.id=d.scan_id "
            f"WHERE {' AND '.join(cond)}", params).fetchone()[0]


def stats() -> dict:
    with closing(db()) as con:
        per_class = [dict(r) for r in con.execute(
            "SELECT label, COUNT(*) AS count FROM detections GROUP BY label ORDER BY count DESC")]
        hist = {r[0]: r[1] for r in con.execute(
            "SELECT MIN(CAST(confidence*10 AS INTEGER), 9) b, COUNT(*) FROM detections GROUP BY b")}
        daily = [dict(r) for r in con.execute(
            "SELECT substr(ts,1,10) AS date, COUNT(*) AS count FROM scans "
            "GROUP BY date ORDER BY date DESC LIMIT 14")][::-1]
        cond = {r[0]: r[1] for r in con.execute(
            "SELECT condition, COUNT(*) FROM detections GROUP BY condition")}
        total_scans = con.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        total_obj = con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        n_feedback = con.execute("SELECT COUNT(*) FROM waste_log").fetchone()[0]
    return {"total_scans": total_scans, "total_objects": total_obj, "feedback_rows": n_feedback,
            "per_class": per_class, "daily": daily,
            "conditions": {"fresh": cond.get("fresh", 0), "rotten": cond.get("rotten", 0)},
            "confidence_hist": [{"bin": f"{i/10:.1f}-{(i+1)/10:.1f}", "count": hist.get(i, 0)}
                                for i in range(10)]}


EXPORT_COLS = ["scan_id", "ts", "filename", "label", "fruit", "condition", "confidence",
               "x1", "y1", "x2", "y2"]
EXPORT_META = {
    "csv": ("text/csv", "deteksi.csv"),
    "xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "deteksi.xlsx"),
    "json": ("application/json", "deteksi.json"),
}
FORMULA_PREFIX = ("=", "+", "-", "@", "\t", "\r")


def csv_safe(value):
    """Cegah formula injection saat CSV dibuka di Excel/LibreOffice."""
    if isinstance(value, str) and value.startswith(FORMULA_PREFIX):
        return "'" + value
    return value


def export_bytes(fmt: str, rows: list[dict]) -> tuple[bytes, str, str]:
    """Kembalikan (isi, mime, nama_file) untuk csv, json, atau xlsx."""
    if fmt == "json":
        return (json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"),
                "application/json", "deteksi.json")
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(EXPORT_COLS)
        w.writerows([[csv_safe(r[c]) for c in EXPORT_COLS] for r in rows])
        return ("\ufeff" + buf.getvalue()).encode("utf-8"), "text/csv", "deteksi.csv"
    if fmt == "xlsx":
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Deteksi"
        ws.append(EXPORT_COLS)
        for r in rows:
            ws.append([r[c] for c in EXPORT_COLS])
        for row in ws.iter_rows(min_row=2):  # paksa teks menjadi string, bukan formula
            for cell in row:
                if isinstance(cell.value, str):
                    cell.data_type = "s"
        out = io.BytesIO()
        wb.save(out)
        return (out.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "deteksi.xlsx")
    raise AppError("format harus csv, json, atau xlsx")


def deferred_export(fmt: str, label=None, min_conf=0.0,
                    sort_by="timestamp", order="desc") -> Callable[[], bytes]:
    """Callable untuk st.download_button: file dibuat hanya saat diklik.

    Tanpa ini Streamlit mengambil data ekspor setiap halaman Riwayat dirender,
    bahkan saat tombol unduh belum disentuh.
    """
    if fmt not in EXPORT_META:
        raise AppError("format harus csv, json, atau xlsx")

    def build() -> bytes:
        return export_bytes(fmt, det_query(label, min_conf, sort_by, order))[0]

    return build


# ------------------------------------------------- log retur / koreksi staf
def save_feedback(scan_id: int, status: str, items: list[dict]) -> str:
    """Satu keputusan per scan. Mencegah satu objek tercatat ganda."""
    if status not in ACTION_BY_STATUS:
        raise AppError("Status koreksi tidak valid")
    if not items or len(items) > 50:
        raise AppError("Jumlah item harus 1 sampai 50")
    for i in items:
        f, c = str(i.get("fruit", "")).strip(), i.get("confidence")
        if not 1 <= len(f) <= 50 or not isinstance(c, (int, float)) or not 0 <= c <= 1:
            raise AppError("Data item tidak valid")
    action = ACTION_BY_STATUS[status]
    with closing(db()) as con, con:
        # Kunci tulis sejak awal agar dua proses paralel tidak lolos cek duplikat.
        con.execute("BEGIN IMMEDIATE")
        if not con.execute("SELECT 1 FROM scans WHERE id=?", (scan_id,)).fetchone():
            raise AppError("scan_id tidak ditemukan")
        if con.execute("SELECT 1 FROM waste_log WHERE scan_id=?", (scan_id,)).fetchone():
            raise AppError("Scan ini sudah punya keputusan staf")
        ts = now()
        con.executemany(
            "INSERT INTO waste_log(scan_id,ts,fruit,status,confidence,action) VALUES(?,?,?,?,?,?)",
            [(scan_id, ts, str(i["fruit"]).strip().capitalize(), status, float(i["confidence"]), action)
             for i in items])
    if status != "Rotten (Afkir)":  # koreksi staf = data latih berharga
        try:  # gambar opsional; jangan gagalkan log yang sudah tersimpan
            src = RECENT_DIR / f"{scan_id}.jpg"
            if src.exists():
                REVIEW_DIR.mkdir(parents=True, exist_ok=True)
                tag = "fp" if status.startswith("False Positive") else "fn"
                shutil.copy(src, REVIEW_DIR / f"{scan_id}_{tag}.jpg")
        except OSError:
            pass
    return action


def waste_log(limit: int = 200) -> list[dict]:
    with closing(db()) as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM waste_log ORDER BY id DESC LIMIT ?", (limit,))]


def waste_log_csv() -> bytes:
    """Format kolom sama dengan waste_log.csv versi Streamlit lama."""
    with closing(db()) as con:
        rows = con.execute("SELECT * FROM waste_log ORDER BY id").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Timestamp", "Jenis Buah", "Status Deteksi", "Confidence", "Tindakan Sistem/Petugas"])
    for r in rows:
        w.writerow([csv_safe(r["ts"]), csv_safe(r["fruit"]), csv_safe(r["status"]),
                    f"{r['confidence']:.2%}", csv_safe(r["action"])])
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


# ------------------------------------------------------------ penjualan (kasir)
def next_receipt_no() -> int:
    with closing(db()) as con:
        return (con.execute("SELECT MAX(id) FROM sales").fetchone()[0] or 0) + 1


def save_sale(total: int, qty: int, discount_pct: int, cash: int, items: list[dict]) -> int:
    with closing(db()) as con, con:
        cur = con.execute(
            "INSERT INTO sales(ts,total,qty,discount_pct,cash,items) VALUES(?,?,?,?,?,?)",
            (now(), total, qty, discount_pct, cash, json.dumps(items, ensure_ascii=False)))
        return cur.lastrowid


def today_sales() -> dict:
    with closing(db()) as con:
        n, s = con.execute("SELECT COUNT(*), COALESCE(SUM(total),0) FROM sales WHERE substr(ts,1,10)=?",
                           (now()[:10],)).fetchone()
    return {"n": n, "sum": s}


def receipt_text(no: int, ts: str, cart: list[dict], afkir: dict, t: dict) -> str:
    W = 34

    def two(a: str, b: str) -> str:
        return a + " " * max(1, W - len(a) - len(b)) + b

    lines = ["SMART QC BUAH".center(W), f"No. {no:04d}".center(W), ts.center(W), "-" * W]
    for c in cart:
        lines.append(two(nama(c["fruit"]), rp(c["price"] * c["qty"])))
        lines.append(f"  {c['qty']} x {rp(c['price'])}")
    lines.append("-" * W)
    if t["disc"]:
        lines += [two("Subtotal", rp(t["sub"])), two(f"Diskon {t['d']}%", "-" + rp(t["disc"]))]
    lines.append(two("TOTAL", rp(t["total"])))
    if t["cash"]:
        lines += [two("Tunai", rp(t["cash"])), two("Kembalian", rp(max(0, t["change"])))]
    if afkir:
        lines += ["-" * W, "Tidak dihitung (busuk):"]
        lines += [f"  {nama(f)} x{n}" for f, n in afkir.items()]
    lines += ["-" * W, "Terima kasih".center(W)]
    return "\n".join(lines)


# ------------------------------------------------------ kamera realtime (WebRTC)
def draw_boxes(img, objects) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    for o in objects:
        x1, y1, x2, y2 = (int(v) for v in o["bbox"])
        rotten = o["condition"] == "rotten"
        col = (68, 68, 239) if rotten else (94, 197, 34)  # BGR dari #ef4444 dan #22c55e
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
        text = f"{nama(o['fruit'])} {'busuk' if rotten else rp(o['price'])}"
        (tw, _), _ = cv2.getTextSize(text, font, 0.55, 1)
        top = max(0, y1 - 22)
        cv2.rectangle(img, (x1, top), (x1 + tw + 10, top + 22), col, -1)
        cv2.putText(img, text, (x1 + 5, max(16, y1 - 6)), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


class LiveProcessor(VideoProcessorBase):
    """Memproses setiap frame kamera di thread terpisah. Pratinjau live tidak disimpan ke database."""

    def __init__(self):
        self._lock = threading.Lock()
        self.conf, self.iou, self.classes = 0.5, 0.4, None
        self.raw = None
        self.fps, self.latency, self.n, self.bad, self.error = 0.0, 0.0, 0, 0, None
        self._last = time.perf_counter()

    def configure(self, conf, iou, classes) -> None:
        with self._lock:
            self.conf, self.iou, self.classes = conf, iou, classes

    def snapshot(self):
        with self._lock:
            return None if self.raw is None else self.raw.copy()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        raw = cv2.resize(img, (1280, round(h * 1280 / w))) if w > 1280 else img
        small = cv2.resize(img, (640, round(h * 640 / w))) if w != 640 else img.copy()
        with self._lock:
            self.raw = raw
            conf, iou, classes = self.conf, self.iou, self.classes
        try:
            r = detect_image(small, conf, iou, classes, annotate=False)
            draw_boxes(small, r["objects"])
            t = time.perf_counter()
            inst = 1.0 / max(t - self._last, 1e-6)
            self.fps = inst if self.fps == 0 else 0.8 * self.fps + 0.2 * inst
            self._last = t
            self.latency, self.n = r["latency_ms"], r["n_objects"]
            self.bad = sum(o["condition"] == "rotten" for o in r["objects"])
            self.error = None
        except AppError as e:
            self.error = str(e)
        except Exception as e:  # jangan matikan stream karena satu frame gagal
            self.error = f"Galat deteksi: {e}"
        return av.VideoFrame.from_ndarray(small, format="bgr24")


@st.fragment(run_every=1)
def live_info(ctx, conf, iou, classes) -> None:
    """Segarkan FPS dan ringkasan tiap detik tanpa menjalankan ulang seluruh halaman."""
    proc = ctx.video_processor
    if proc is None:
        st.caption("FPS: - | Latensi: - ms")
        return
    proc.configure(conf, iou, classes)
    if proc.error:
        st.caption(proc.error)
        return
    st.caption(f"FPS: {proc.fps:.1f} | Latensi: {proc.latency:.0f} ms")
    if proc.n:
        st.caption(f"{proc.n} buah terlihat, {proc.bad} busuk. Tekan Ambil dan catat untuk memproses.")
    else:
        st.caption("Belum ada buah terlihat. Dekatkan buah ke kamera.")


# ----------------------------------------------------------- state dan keranjang
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("cart", [])
    ss.setdefault("afkir", {})
    ss.setdefault("prices", dict(HARGA_BUAH))
    ss.setdefault("results", [])
    ss.setdefault("up_key", 0)
    ss.setdefault("hist_off", 0)
    ss.setdefault("receipt", None)
    ss.setdefault("confirm", None)
    ss.setdefault("cam_hash", None)


def cart_mode() -> bool:
    return st.session_state.get("mode", MODE_CART) == MODE_CART


def add_to_cart(objs: list[dict]) -> int:
    ss, n = st.session_state, 0
    for o in objs:
        if o["condition"] == "rotten":
            ss.afkir[o["fruit"]] = ss.afkir.get(o["fruit"], 0) + 1
            continue
        price = o.get("price") or ss.prices.get(o["fruit"]) or DEFAULT_PRICE
        ss.prices[o["fruit"]] = price
        item = next((c for c in ss.cart if c["fruit"] == o["fruit"]), None)
        if item:
            item["qty"] += 1
        else:
            ss.cart.append({"fruit": o["fruit"], "price": price, "qty": 1})
        n += 1
    return n


def calc() -> dict:
    ss = st.session_state
    sub = sum(c["qty"] * c["price"] for c in ss.cart)
    d = min(100, max(0, int(ss.get("disc", 0) or 0)))
    disc = (sub * d + 50) // 100
    total = sub - disc
    cash = int(ss.get("cash", 0) or 0)
    return {"sub": sub, "d": d, "disc": disc, "total": total, "cash": cash,
            "change": cash - total, "short": 0 < cash < total}


# callback tombol (dijalankan sebelum skrip dirender ulang, aman mengubah state widget)
def cb_qty(fruit: str, d: int) -> None:
    cart = st.session_state.cart
    for i, c in enumerate(cart):
        if c["fruit"] == fruit:
            if d == 0 or c["qty"] + d < 1:
                cart.pop(i)
            else:
                c["qty"] += d
            return


def cb_cash(amount: int) -> None:
    st.session_state.cash = amount or calc()["total"]


def cb_manual() -> None:
    ss = st.session_state
    f = ss.get("man_sel")
    if f:
        add_to_cart([{"fruit": f, "condition": "fresh", "price": ss.prices.get(f, DEFAULT_PRICE)}])
        st.toast(f"{nama(f)} ditambahkan manual")


def cb_ask(kind: str) -> None:
    st.session_state.confirm = kind


def cb_cancel() -> None:
    st.session_state.confirm = None


def cb_clear() -> None:
    ss = st.session_state
    ss.cart, ss.afkir, ss.confirm = [], {}, None


def cb_pay() -> None:
    ss = st.session_state
    t, qty = calc(), sum(c["qty"] for c in ss.cart)
    if not ss.cart or t["short"]:
        ss.confirm = None
        return
    no = next_receipt_no()
    text = receipt_text(no, now(), ss.cart, ss.afkir, t)
    save_sale(t["total"], qty, t["d"], t["cash"], ss.cart)
    ss.receipt = {"no": no, "text": text}
    ss.cart, ss.afkir, ss.confirm = [], {}, None
    ss.disc, ss.cash = 0, 0
    st.toast("Transaksi selesai")


def find_result(scan_id: int):
    return next((r for r in st.session_state.results if r.get("scan_id") == scan_id), None)


def cb_add_result(scan_id: int) -> None:
    r = find_result(scan_id)
    if r and not r["added"]:
        r["added"] = True
        st.toast(f"{add_to_cart(r['objects'])} buah masuk keranjang")


def cb_feedback(scan_id: int, status: str, which: str) -> None:
    r = find_result(scan_id)
    if not r:
        return
    objs = [o for o in r["objects"] if o["condition"] == "rotten"] if which == "rotten" else r["objects"]
    items = [{"fruit": o["fruit"], "confidence": o["confidence"]} for o in objs]
    try:
        action = save_feedback(scan_id, status, items)
    except AppError as e:
        r["fb"] = ("err", str(e))
        return
    r["fb"] = ("ok", f"Tercatat: {action}")
    if status.startswith("False Positive") and cart_mode() and not r["added"]:
        r["added"] = True  # koreksi staf: buah layak jual, hitung harganya
        fixed = [{**o, "condition": "fresh", "price": st.session_state.prices.get(o["fruit"], DEFAULT_PRICE)}
                 for o in objs]
        st.toast(f"{add_to_cart(fixed)} buah dikoreksi dan masuk keranjang")


def handle_bytes(data: bytes, name: str, source: str, s: dict) -> None:
    """Deteksi satu gambar, simpan, dan taruh kartu hasilnya di urutan teratas."""
    ss = st.session_state
    try:
        r = run_detection(data, name, source, s["conf"], s["iou"], s["classes"])
        r.update(kind="ok", added=False, fb=None)
        if cart_mode() and r["status"] == "fresh" and s["auto"]:
            add_to_cart(r["objects"])
            r["added"] = True
    except AppError as e:
        r = {"kind": "err", "filename": name, "error": str(e)}
    ss.results.insert(0, r)
    del ss.results[MAX_RESULT_CARDS:]


# --------------------------------------------------------------------- UI: sidebar
def sidebar() -> dict:
    sb = st.sidebar
    sb.title("Smart QC Buah")
    sb.caption("Kasir deteksi buah")
    if not Path(MODEL_PATH).exists():
        sb.error("Model tidak ditemukan")
        sb.caption(f"Letakkan bobot YOLOv8 di `{MODEL_PATH}`")
    elif model_store().loaded:
        sb.success("Aplikasi dan model siap")
    else:
        sb.warning("Model dimuat saat scan pertama")
    sb.subheader("Pengaturan deteksi")
    conf = sb.slider("Confidence minimum", 0.05, 0.95, 0.50, 0.05)
    iou = sb.slider("IoU (NMS)", 0.10, 0.90, 0.40, 0.05)
    picked = sb.multiselect("Jenis buah yang dikenali", FRUITS, default=FRUITS, format_func=nama)
    if not picked:
        sb.caption("Tidak ada yang dipilih, semua jenis dipakai.")
        picked = FRUITS
    auto = sb.checkbox("Masukkan ke keranjang otomatis jika semua buah segar", value=True)
    return {"conf": conf, "iou": iou, "auto": auto,
            "classes": None if len(picked) == len(FRUITS) else picked}


# --------------------------------------------------------------------- UI: kasir
def render_result(r: dict) -> None:
    with st.container(border=True):
        if r["kind"] == "err":
            st.markdown(f"**{r['filename']}**")
            st.error(f"Gagal: {r['error']}")
            return
        sid = r["scan_id"]
        left, right = st.columns([1, 1.1])
        if r.get("annotated_rgb") is not None:
            left.image(r["annotated_rgb"])
        right.markdown(f"**{r['filename']}**  :{STATUS_COLOR[r['status']]}[{STATUS_TXT[r['status']]}]")
        if r["status"] == "fresh":
            right.markdown(f"### {rp(r['total_price'])}")
            right.caption(f"{r['n_objects']} buah")
        if r["objects"]:
            right.dataframe(pd.DataFrame([{
                "Buah": nama(o["fruit"]),
                "Kondisi": "busuk" if o["condition"] == "rotten" else "segar",
                "Yakin (%)": round(o["confidence"] * 100),
                "Harga": "Retur" if o["condition"] == "rotten" else rp(o["price"]),
            } for o in r["objects"]]), hide_index=True)
        if cart_mode() and r["n_objects"] > 0:
            right.button("Sudah di keranjang" if r["added"] else "Tambah ke keranjang",
                         key=f"add_{sid}", disabled=r["added"], type="primary",
                         on_click=cb_add_result, args=(sid,))
        done = bool(r["fb"] and r["fb"][0] == "ok")
        if r["status"] == "rotten":
            right.button("Konfirmasi afkir", key=f"fb1_{sid}", disabled=done,
                         on_click=cb_feedback, args=(sid, "Rotten (Afkir)", "rotten"))
            right.button("Sistem salah, buah layak jual", key=f"fb2_{sid}", disabled=done,
                         on_click=cb_feedback, args=(sid, "False Positive (Koreksi Staf)", "rotten"))
        elif r["status"] == "fresh":
            right.button("Ada kerusakan yang terlewat", key=f"fb3_{sid}", disabled=done,
                         on_click=cb_feedback, args=(sid, "False Negative (Koreksi Staf)", "all"))
        if r["fb"]:
            (right.success if r["fb"][0] == "ok" else right.error)(r["fb"][1])
        right.caption(f"Latensi model {r['latency_ms']} ms")


def input_panel(s: dict) -> None:
    ss = st.session_state
    c1, c2 = st.columns(2)
    src = c1.radio("Sumber", ["Upload foto", "Kamera"], horizontal=True, key="src")
    c2.radio("Mode", [MODE_CART, MODE_CHECK], horizontal=True, key="mode",
             help="Cek harga saja tidak menambah ke keranjang")

    if src == "Upload foto":
        files = st.file_uploader(
            "Seret foto belanjaan ke sini (JPG/PNG, maks 10 MB, boleh banyak file)",
            type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"up_{ss.up_key}")
        if files:
            if len(files) > MAX_BATCH:
                st.warning(f"Maksimal {MAX_BATCH} file sekali proses, sisanya diabaikan.")
                files = files[:MAX_BATCH]
            with st.spinner("Memproses gambar ..."):
                for f in files:
                    handle_bytes(f.getvalue(), f.name, "upload", s)
            ss.up_key += 1  # kosongkan uploader agar file yang sama bisa dipilih lagi
            st.rerun()
        return

    if not HAS_WEBRTC:
        st.info("Paket streamlit-webrtc belum terpasang, kamera realtime tidak tersedia. "
                "Memakai mode foto tunggal. Pasang dengan `pip install streamlit-webrtc`.")
        shot = st.camera_input("Ambil foto buah")
        if shot is not None:
            data = shot.getvalue()
            h = hashlib.md5(data).hexdigest()
            if h != ss.cam_hash:  # satu foto diproses satu kali meski skrip dijalankan ulang
                ss.cam_hash = h
                handle_bytes(data, f"kamera-{int(time.time() * 1000)}.jpg", "camera", s)
        return

    ctx = webrtc_streamer(
        key="live", video_processor_factory=LiveProcessor, async_processing=True,
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
                                  "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    if ctx.video_processor:
        ctx.video_processor.configure(s["conf"], s["iou"], s["classes"])
    live_info(ctx, s["conf"], s["iou"], s["classes"])
    st.caption("Kotak hijau berarti segar, merah berarti busuk. Pratinjau live tidak dicatat.")
    if st.button("Ambil dan catat", disabled=not ctx.state.playing, type="primary", key="shot"):
        frame = ctx.video_processor.snapshot() if ctx.video_processor else None
        if frame is None:
            st.toast("Kamera belum siap")
        else:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            handle_bytes(buf.tobytes(), f"kamera-{int(time.time() * 1000)}.jpg", "camera", s)


def receipt_panel() -> None:
    ss = st.session_state
    ss.setdefault("disc", 0)
    ss.setdefault("cash", 0)
    with st.container(border=True):
        st.subheader("Struk belanja")
        st.caption(f"No. {next_receipt_no():04d} | {datetime.now():%d-%m-%Y %H:%M}")
        if not ss.cart:
            st.info("Keranjang kosong. Foto belanjaan pelanggan atau nyalakan kamera untuk mulai.")
        for c in ss.cart:
            f = c["fruit"]
            a, b = st.columns([3, 2])
            a.markdown(f"**{nama(f)}**")
            b.markdown(f"**{rp(c['price'] * c['qty'])}**")
            q1, q2, q3, q4 = st.columns([1, 2.4, 1, 1.8])
            q1.button("-", key=f"minus_{f}", on_click=cb_qty, args=(f, -1))
            q2.caption(f"{c['qty']} x {rp(c['price'])}")
            q3.button("+", key=f"plus_{f}", on_click=cb_qty, args=(f, 1))
            q4.button("Hapus", key=f"del_{f}", on_click=cb_qty, args=(f, 0))

        t = calc()
        if t["disc"]:
            st.markdown(f"Subtotal: {rp(t['sub'])}  \nDiskon {t['d']}%: -{rp(t['disc'])}")
        st.markdown(f"## TOTAL {rp(t['total'])}")
        if t["cash"]:
            if t["short"]:
                st.error(f"Uang kurang {rp(abs(t['change']))}")
            else:
                st.success(f"Kembalian {rp(t['change'])}")
        qty = sum(c["qty"] for c in ss.cart)
        if qty:
            st.caption(f"{qty} buah, {len(ss.cart)} jenis")
        if ss.afkir:
            st.error("Tidak dihitung (busuk): "
                     + ", ".join(f"{nama(f)} x{n}" for f, n in ss.afkir.items())
                     + ". Ganti dengan buah segar atau retur.")

        m1, m2 = st.columns([2, 1.2])
        price_snap = dict(ss.prices)  # salinan lokal: format_func tidak boleh membaca session_state
        m1.selectbox("Tambah manual", FRUITS, key="man_sel", label_visibility="collapsed",
                     format_func=lambda f: f"{nama(f)} - {rp(price_snap.get(f, DEFAULT_PRICE))}")
        m2.button("Tambah manual", on_click=cb_manual,
                  help="Pakai jika kamera atau model tidak mengenali buah")
        st.number_input("Diskon (%)", min_value=0, max_value=100, step=1, key="disc")
        st.number_input("Uang diterima (Rp)", min_value=0, step=500, key="cash")
        q1, q2, q3 = st.columns(3)
        q1.button("Uang pas", on_click=cb_cash, args=(0,))
        q2.button("50.000", on_click=cb_cash, args=(50000,))
        q3.button("100.000", on_click=cb_cash, args=(100000,))

        if ss.confirm == "pay":
            st.warning("Transaksi selesai? Keranjang akan dikosongkan.")
            y, n = st.columns(2)
            y.button("Ya, selesai", key="pay_yes", type="primary", on_click=cb_pay)
            n.button("Batal", key="pay_no", on_click=cb_cancel)
        elif ss.confirm == "clear":
            st.warning("Kosongkan keranjang?")
            y, n = st.columns(2)
            y.button("Ya, kosongkan", key="clr_yes", type="primary", on_click=cb_clear)
            n.button("Batal", key="clr_no", on_click=cb_cancel)
        else:
            st.button("Selesai dan simpan struk", type="primary", disabled=not ss.cart or t["short"],
                      on_click=cb_ask, args=("pay",))
            st.button("Kosongkan keranjang", disabled=not ss.cart and not ss.afkir,
                      on_click=cb_ask, args=("clear",))

        if ss.receipt:
            st.success(f"Transaksi No. {ss.receipt['no']:04d} selesai")
            st.code(ss.receipt["text"], language=None)
            text = ss.receipt["text"]
            st.download_button("Unduh struk (.txt)", lambda: text.encode("utf-8"),
                               file_name=f"struk-{ss.receipt['no']:04d}.txt", mime="text/plain",
                               on_click="ignore")


def page_kasir(s: dict) -> None:
    left, right = st.columns([1.5, 1], gap="medium")
    with left:
        with st.container(border=True):
            input_panel(s)
        for r in st.session_state.results:
            render_result(r)
    with right:
        receipt_panel()


# ------------------------------------------------------------- UI: halaman lain
def page_harga() -> None:
    st.subheader("Daftar harga buah")
    st.caption("Harga per buah yang dipakai sistem. Buah busuk tidak dihitung (Rp 0) dan harus diretur.")
    cols = st.columns(len(FRUITS))
    for col, f in zip(cols, FRUITS):
        with col.container(border=True):
            st.metric(nama(f), rp(st.session_state.prices.get(f, DEFAULT_PRICE)))
            st.caption("per buah")


def page_dashboard() -> None:
    s, sales = stats(), today_sales()
    tot = s["conditions"]["fresh"] + s["conditions"]["rotten"]
    pct = f"{round(s['conditions']['rotten'] * 100 / tot)}%" if tot else "-"
    kpis = [("Total scan", s["total_scans"]), ("Total objek", s["total_objects"]),
            ("Segar", s["conditions"]["fresh"]), ("Busuk", s["conditions"]["rotten"]),
            ("Persentase busuk", pct), ("Koreksi dan retur", s["feedback_rows"]),
            ("Transaksi hari ini", sales["n"]), ("Omzet hari ini", rp(sales["sum"]))]
    for row in (kpis[:4], kpis[4:]):
        for col, (k, v) in zip(st.columns(4), row):
            col.metric(k, v)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Objek per kelas**")
        if s["per_class"]:
            st.bar_chart(pd.DataFrame(s["per_class"]).set_index("label"))
        else:
            st.caption("Belum ada data.")
    with c2:
        st.markdown("**Distribusi confidence**")
        st.bar_chart(pd.DataFrame(s["confidence_hist"]).set_index("bin"))
    st.markdown("**Scan per hari (14 hari terakhir)**")
    if s["daily"]:
        st.line_chart(pd.DataFrame(s["daily"]).set_index("date"))
    else:
        st.caption("Belum ada data.")


def reset_hist_offset() -> None:
    st.session_state.hist_off = 0


def cb_page(delta: int) -> None:
    st.session_state.hist_off = max(0, st.session_state.hist_off + delta)


def page_riwayat() -> None:
    ss, lim = st.session_state, 25
    st.subheader("Riwayat scan")
    c1, c2, c3 = st.columns(3)
    sort_by = c1.selectbox("Urut", list(SCAN_SORT), key="h_sort", on_change=reset_hist_offset,
                           format_func={"timestamp": "Waktu", "n_objects": "Jumlah objek",
                                        "latency": "Latensi", "status": "Status"}.get)
    order = c2.selectbox("Arah", ["desc", "asc"], key="h_order", on_change=reset_hist_offset,
                         format_func={"desc": "Turun", "asc": "Naik"}.get)
    status = c3.selectbox("Status", ["", "fresh", "rotten", "none"], key="h_status",
                          on_change=reset_hist_offset, format_func=lambda x: x or "Semua")
    j = history(sort_by, order, status or None, lim, ss.hist_off)
    if j["items"]:
        df = pd.DataFrame(j["items"]).rename(columns={
            "id": "ID", "ts": "Waktu", "filename": "File", "source": "Sumber",
            "n_objects": "Objek", "status": "Status", "latency_ms": "Latensi (ms)"})
        st.dataframe(df, hide_index=True)
    else:
        st.caption("Belum ada scan.")
    p1, p2, p3 = st.columns([1, 2, 1])
    p1.button("Sebelumnya", disabled=ss.hist_off == 0, on_click=cb_page, args=(-lim,))
    if j["total"]:
        p2.caption(f"{ss.hist_off + 1}-{min(ss.hist_off + lim, j['total'])} dari {j['total']}")
    p3.button("Berikutnya", disabled=ss.hist_off + lim >= j["total"], on_click=cb_page, args=(lim,))

    with st.expander("Ekspor deteksi"):
        with closing(db()) as con:
            labels = [r[0] for r in con.execute("SELECT DISTINCT label FROM detections ORDER BY label")]
        e1, e2, e3, e4 = st.columns(4)
        label = e1.selectbox("Label atau buah", [""] + FRUITS + labels, format_func=lambda x: x or "Semua")
        min_conf = e2.slider("Confidence minimum", 0.0, 1.0, 0.0, 0.05, key="e_conf")
        e_sort = e3.selectbox("Urut", list(DET_SORT), key="e_sort")
        e_order = e4.selectbox("Arah", ["desc", "asc"], key="e_order")
        st.caption(f"{det_count(label or None, min_conf)} baris akan diekspor.")
        d1, d2, d3 = st.columns(3)
        for col, fmt, text in ((d1, "csv", "Ekspor CSV"), (d2, "xlsx", "Ekspor Excel"), (d3, "json", "Ekspor JSON")):
            mime, fname = EXPORT_META[fmt]
            col.download_button(text, deferred_export(fmt, label or None, min_conf, e_sort, e_order),
                                file_name=fname, mime=mime, key=f"dl_{fmt}", on_click="ignore")


def page_log() -> None:
    st.subheader("Barang afkir dan koreksi staf")
    c1, c2 = st.columns([3, 1])
    q = c1.text_input("Cari", placeholder="Cari buah atau tindakan", label_visibility="collapsed").lower()
    c2.download_button("Unduh laporan retur CSV", waste_log_csv, file_name="laporan_retur_supplier.csv",
                       mime="text/csv", on_click="ignore")
    rows = [r for r in waste_log() if not q or q in " ".join([r["fruit"], r["status"], r["action"]]).lower()]
    if rows:
        st.dataframe(pd.DataFrame([{
            "Waktu": r["ts"], "Buah": r["fruit"], "Status": r["status"],
            "Yakin": f"{r['confidence'] * 100:.1f}%", "Tindakan": r["action"]} for r in rows]),
            hide_index=True)
    else:
        st.caption("Belum ada catatan.")


# ----------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="Smart QC Buah - Kasir", layout="wide")
    ensure_db()
    init_state()
    s = sidebar()
    page = st.radio("Menu", PAGES, horizontal=True, key="page", label_visibility="collapsed")
    if page == "Kasir":
        page_kasir(s)
    elif page == "Daftar harga":
        page_harga()
    elif page == "Dashboard":
        page_dashboard()
    elif page == "Riwayat":
        page_riwayat()
    else:
        page_log()


if __name__ == "__main__":
    main()
