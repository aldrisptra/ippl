# app.py (updated)
import os, re, cv2, json, time, httpx, base64, asyncio, traceback, io
import numpy as np
from typing import Optional, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO

# === Path absolut ke folder file ini ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# Konfigurasi yang bisa diubah
# =========================
YOLO_MODEL_PATH = "yolov8n.pt"   # auto-download pertama kali
YOLO_IMGSZ = 352                 # turunkan ke 320 jika CPU berat
YOLO_CONF  = 0.40                # threshold confidence 'person'
IOU_WARN_THRESHOLD = 0.15        # minimal overlap person vs zona waspada
MISSING_WINDOW = 10              # jumlah frame utk moving average "hilang"
ALERT_THRESHOLD = 0.80           # ambang rata-rata skor "hilang" utk ALERT
TELEGRAM_COOLDOWN = 60           # detik antar notifikasi ALERT
WARN_COOLDOWN = 30               # detik antar notifikasi WARN
MIN_PERSON_AREA_RATIO = 0.01     # abaikan bbox person <1% area frame (anti-noise)
# =========================

# Env (opsional) untuk Telegram: baca dari environment dulu, kalau tidak ada pakai placeholder
# Anda bisa set environment variable TG_TOKEN dan TG_CHAT_ID atau TELEGRAM_TOKEN/TELEGRAM_CHAT_ID
TELEGRAM_TOKEN = os.getenv("TG_TOKEN", "7697921487:AAEvZXLkC61Nzx-eh1e2BES1VfqSJ3wN32E")
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID", "1215968232")

# Inisialisasi model (CPU)
model = YOLO(os.path.join(BASE_DIR, YOLO_MODEL_PATH))

app = FastAPI()

# Serve index di root (pakai path absolut)
@app.get("/")
def read_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

# Mount static di /static (jangan mount "/" supaya WS tidak ketutup)
app.mount("/static", StaticFiles(directory=BASE_DIR, html=False), name="static")


def b64_to_ndarray(data_url: str) -> np.ndarray:
    """Decode data URL (image/jpeg base64) menjadi ndarray OpenCV (BGR)."""
    b64 = re.sub(r"^data:image/\w+;base64,", "", data_url)
    img_bytes = base64.b64decode(b64)
    img = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return frame


def iou(a: Dict[str, int], b: Dict[str, int]) -> float:
    """Hitung IoU dua bbox {x,y,w,h}."""
    ax1, ay1, aw, ah = a["x"], a["y"], a["w"], a["h"]
    bx1, by1, bw, bh = b["x"], b["y"], b["w"], b["h"]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def clamp_rect(r: Optional[Dict[str, int]], W: int, H: int) -> Optional[Dict[str, int]]:
    """Pastikan rect ada, size > 0, dan di-clamp ke batas frame."""
    if not r:
        return None
    try:
        x = max(0, int(r["x"])); y = max(0, int(r["y"]))
        w = max(0, int(r["w"])); h = max(0, int(r["h"]))
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    x2 = min(W, x + w); y2 = min(H, y + h)
    if x2 <= x or y2 <= y:
        return None
    return {"x": x, "y": y, "w": x2 - x, "h": y2 - y}


def valid_rect(r: Optional[Dict[str, int]]) -> bool:
    return bool(r and r.get("w", 0) > 0 and r.get("h", 0) > 0)


class State:
    """Simpan konfigurasi & history sederhana untuk deteksi 'kotak hilang'."""
    def __init__(self):
        self.boxROI: Optional[Dict[str, int]] = None
        self.warnZone: Optional[Dict[str, int]] = None
        self.bg_roi_ref: Optional[np.ndarray] = None
        self.missing_score_history = []
        self.last_alert_ts = 0.0
        self.last_warn_ts = 0.0
        self.last_status = "SETUP"

    def reset_bg(self):
        self.bg_roi_ref = None
        self.missing_score_history.clear()


state = State()


async def send_telegram(text: str):
    """Kirim pesan Telegram (opsional)."""
    if not TELEGRAM_TOKEN or "PUT_TOKEN_HERE" in TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or "PUT_CHAT_ID_HERE" in TELEGRAM_CHAT_ID:
        print("Telegram: token/chat_id tidak diset, melewati kirim pesan")
        return  # dilewati kalau token/chat id belum diset
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        print("Telegram error:", e)


async def send_telegram_photo(frame_bgr, caption="Bukti"):
    """Kirim foto (JPEG) ke Telegram secara async."""
    if not TELEGRAM_TOKEN or "PUT_TOKEN_HERE" in TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or "PUT_CHAT_ID_HERE" in TELEGRAM_CHAT_ID:
        print("Telegram: token/chat_id tidak diset, melewati kirim foto")
        return
    try:
        ok, jpg = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            print("Telegram photo: imencode gagal")
            return
        files = {"photo": ("alert.jpg", io.BytesIO(jpg.tobytes()), "image/jpeg")}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        async with httpx.AsyncClient(timeout=20) as client:
            await client.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files)
    except Exception as e:
        print("Telegram photo error:", e)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            obj = json.loads(raw)

            # Terima konfigurasi ROI/zona dari browser
            if obj.get("type") == "config":
                state.boxROI = obj.get("boxROI")
                state.warnZone = obj.get("warnZone")
                if obj.get("reset"):
                    state.reset_bg()  # hanya reset saat diminta
                print("CONFIG RECEIVED (raw):", state.boxROI, state.warnZone, "reset=", obj.get("reset"))
                await ws.send_text(json.dumps({"status": "READY", "info": "Config diterima"}))
                continue

            # Terima frame
            if obj.get("type") != "frame":
                continue

            # Decode frame
            frame = b64_to_ndarray(obj["data"])
            if frame is None:
                await ws.send_text(json.dumps({"status": "ERROR", "info": "Frame decode gagal"}))
                continue

            H, W = frame.shape[:2]

            # Clamp rect berdasarkan ukuran frame SEKARANG
            box = clamp_rect(state.boxROI, W, H)
            zone = clamp_rect(state.warnZone, W, H)

            if not valid_rect(box) or not valid_rect(zone):
                await ws.send_text(json.dumps(
                    {"status": "SETUP", "info": "ROI/Zona belum valid. Tandai ulang."}))
                continue

            # ========== 1) DETEKSI ORANG ==========
            results = model.predict(
                frame, imgsz=YOLO_IMGSZ, classes=[0], conf=YOLO_CONF, verbose=False
            )
            persons = []
            frame_area = W * H
            min_area = MIN_PERSON_AREA_RATIO * frame_area
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    w, h = (x2 - x1), (y2 - y1)
                    if w * h < min_area:
                        continue  # abaikan yang terlalu kecil
                    persons.append({"x": x1, "y": y1, "w": w, "h": h})

            # Cek apakah ada person overlap dengan zona waspada
            warn = False
            if persons:
                for p in persons:
                    if iou(p, zone) >= IOU_WARN_THRESHOLD:
                        warn = True
                        break

            # ========== 2) CEK "KOTAK HILANG" DI ROI ==========
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                await ws.send_text(json.dumps({"status": "SETUP", "info": "ROI kotak di luar frame"}))
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Inisialisasi background reference saat pertama kali
            if state.bg_roi_ref is None:
                state.bg_roi_ref = gray.copy()

            diff = cv2.absdiff(state.bg_roi_ref, gray)
            score = float(np.mean(diff) / 255.0)  # 0..1 (makin besar makin beda)
            # Pemetaan non-linear agar responsif
            missing_score = 1.0 - np.exp(-4 * score)

            state.missing_score_history.append(missing_score)
            if len(state.missing_score_history) > MISSING_WINDOW:
                state.missing_score_history.pop(0)
            avg_missing = float(np.mean(state.missing_score_history))

            # ========== 3) KEPUTUSAN & AKSI ==========
            status, play, say = "NORMAL", None, None
            now = time.time()

            if avg_missing >= ALERT_THRESHOLD:
                status, play = "ALERT", "ALERT"
                say = "Kotak infaq telah dicuri!"
                if now - state.last_alert_ts > TELEGRAM_COOLDOWN:
                    state.last_alert_ts = now
                    asyncio.create_task(send_telegram("⚠️ Kotak infaq telah dicuri!"))
                    # Kirim juga foto bukti (async) jika tersedia
                    try:
                        asyncio.create_task(send_telegram_photo(frame, "⚠️ Kotak infaq terdeteksi HILANG dari kamera!"))
                    except Exception as _:
                        pass
            elif warn:
                status, play = "WARN", "WARN"
                say = "Ada orang yang mendekati kotak infaq."
                if (state.last_status != "WARN") or (now - state.last_warn_ts > WARN_COOLDOWN):
                    state.last_warn_ts = now
                    asyncio.create_task(send_telegram("ℹ️ Ada orang mendekati kotak infaq."))

            state.last_status = status

            await ws.send_text(json.dumps({
                "status": status,
                "play": play,               # bip/sirene (opsional di frontend)
                "say": say,                 # kalimat untuk TTS di browser
                "info": f"person={len(persons)} missing≈{avg_missing:.2f}"
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"status": "ERROR", "info": str(e)}))
        except Exception:
            pass
        print("WS error:", e)
        traceback.print_exc()
