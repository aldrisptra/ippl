import os, time, io, cv2, json, traceback, asyncio
import numpy as np
import httpx
from ultralytics import YOLO

# =========================
# Konfigurasi via ENV
# =========================
MISSING_WINDOW     = int(os.getenv("MISSING_WINDOW",  "20"))     # smoothing hilang-muncul
WARN_THRESHOLD     = float(os.getenv("WARN_THRESHOLD",  "0.60"))
ALERT_THRESHOLD    = float(os.getenv("ALERT_THRESHOLD", "0.90"))
TELEGRAM_COOLDOWN  = int(os.getenv("TELEGRAM_COOLDOWN","10"))    # detik
SHOW_WINDOW        = os.getenv("SHOW_WINDOW", "1") == "1"        # 0 untuk headless
SAVE_ALERT_FRAME   = os.getenv("SAVE_ALERT_FRAME", "1") == "1"
SAVE_DEBUG_NOMISS  = os.getenv("SAVE_DEBUG_NOMISS", "0") == "1"  # simpan frame saat miss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Lokasi default weight → ganti otomatis ke file di root kalau ada
DEFAULT_WEIGHT = os.path.join(BASE_DIR, "runs", "detect", "train_box", "weights", "best.pt")
ROOT_WEIGHT    = os.path.join(BASE_DIR, "best.pt")
if os.path.isfile(ROOT_WEIGHT):
    DEFAULT_WEIGHT = ROOT_WEIGHT

YOLO_WEIGHTS   = os.getenv("YOLO_WEIGHTS", DEFAULT_WEIGHT)
YOLO_CONF      = float(os.getenv("YOLO_CONF", "0.15"))
YOLO_IMG       = int(os.getenv("YOLO_IMG",  "640"))
YOLO_IOU       = float(os.getenv("YOLO_IOU", "0.45"))
MAX_DET        = int(os.getenv("MAX_DET", "20"))

# Filter class (opsional): misal "box" atau "0,2"
CLASS_FILTER   = os.getenv("CLASS_FILTER", "").strip()

# ROI (0..1, opsional), contoh: "0.25,0.30,0.50,0.50"
ROI_STR        = os.getenv("ROI_XYWH", "").strip()

# Minimum luas bbox (rasio terhadap frame area, untuk buang deteksi kecil/noise)
MIN_BOX_AREA_P = float(os.getenv("MIN_BOX_AREA_P", "0.0010"))  # 0.1% frame area

# Kamera
CAM_INDEX      = int(os.getenv("CAM_INDEX", "0"))
CAM_BACKEND    = os.getenv("CAM_BACKEND", "MSMF").upper()      # MSMF | DSHOW | ANY
CAP_MAP = {
    "MSMF": cv2.CAP_MSMF,
    "DSHOW": cv2.CAP_DSHOW,
    "ANY": cv2.CAP_ANY,
}

# Telegram (opsional)
TELEGRAM_TOKEN   = os.getenv("TG_TOKEN",   "7697921487:AAEvZXLkC61Nzx-eh1e2BES1VfqSJ3wN32E")
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID", "1215968232")


# =========================
# Telegram helper
# =========================
async def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG] token/chat_id belum diset, skip.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        print("[TG] error:", e)

async def send_telegram_photo(frame_bgr, caption="Bukti"):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG] token/chat_id belum diset, skip foto.")
        return
    try:
        ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            print("[TG] photo: imencode gagal")
            return
        files = {"photo": ("alert.jpg", io.BytesIO(jpg.tobytes()), "image/jpeg")}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files)
    except Exception as e:
        print("[TG] photo error:", e)


# =========================
# Util
# =========================
def draw_status(frame, text, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def parse_roi(h, w, roi_str: str):
    """Return (x1,y1,x2,y2) pixel ROI if valid, else None."""
    if not roi_str:
        return None
    try:
        fx, fy, fw, fh = [float(s) for s in roi_str.split(",")]
        fx = np.clip(fx, 0, 1); fy = np.clip(fy, 0, 1)
        fw = np.clip(fw, 0, 1); fh = np.clip(fh, 0, 1)
        x1 = int(fx * w)
        y1 = int(fy * h)
        x2 = int(min(w, (fx + fw) * w))
        y2 = int(min(h, (fy + fh) * h))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)
    except Exception:
        return None

def open_camera(index: int, backend_key: str):
    cap = cv2.VideoCapture(index, CAP_MAP.get(backend_key, cv2.CAP_ANY))
    if not cap.isOpened() and backend_key != "ANY":
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    return cap

def parse_class_filter(cf_raw: str, names: dict):
    """Terima '0,2' atau 'box,other' → kembalikan list id class (int)."""
    if not cf_raw:
        return None
    cf_raw = cf_raw.strip()
    ids = []
    by_name = []
    # coba split dengan koma
    parts = [s.strip() for s in cf_raw.split(",") if s.strip()]
    for p in parts:
        if p.isdigit():
            ids.append(int(p))
        else:
            by_name.append(p.lower())
    if by_name:
        # map nama → id
        name_to_id = {str(v).lower(): int(k) for k, v in names.items()}
        for nm in by_name:
            if nm in name_to_id:
                ids.append(name_to_id[nm])
    # unik dan urut
    return sorted(set(ids)) if ids else None


# =========================
# Main
# =========================
def main():
    # ====== Cek & Load model ======
    if not os.path.isfile(YOLO_WEIGHTS):
        print(f"[YOLO] Weights tidak ditemukan: {YOLO_WEIGHTS}")
        print("       Set env YOLO_WEIGHTS ke lokasi best.pt kamu.")
        return

    try:
        model = YOLO(YOLO_WEIGHTS)
        # Paksa CPU (aman di mesin tanpa CUDA)
        device = "cpu"
        model.to(device)
        print("[YOLO] loaded:", YOLO_WEIGHTS)
        print("[YOLO] device:", device)
        print("[YOLO] classes:", model.names)
    except Exception as e:
        print("[YOLO] gagal load (mungkin file korup):", e)
        print("       Coba ganti YOLO_WEIGHTS ke file best.pt lain yang sukses ter-validate.")
        return

    # Kunci kelas bila di-set
    class_ids = parse_class_filter(CLASS_FILTER, model.names)
    if class_ids:
        print("[CFG] CLASS_FILTER aktif →", class_ids, "(nama:", [model.names[i] for i in class_ids], ")")

    # ====== Buka kamera ======
    cap = open_camera(CAM_INDEX, CAM_BACKEND)
    if not cap.isOpened():
        print(f"[CAM] gagal membuka kamera index {CAM_INDEX} (backend {CAM_BACKEND})")
        return

    # Set resolusi (opsional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    absent_hist   = []
    last_alert_ts = 0.0
    last_status   = "INIT"
    frame_count   = 0

    print(f"[INFO] tekan 'q' untuk keluar. CONF={YOLO_CONF:.2f} IMG={YOLO_IMG} IOU={YOLO_IOU:.2f} BACKEND={CAM_BACKEND}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[CAM] gagal membaca frame")
                time.sleep(0.05)
                continue

            H, W     = frame.shape[:2]
            roi_px   = parse_roi(H, W, ROI_STR)
            infer_src= frame if not roi_px else frame[roi_px[1]:roi_px[3], roi_px[0]:roi_px[2]].copy()

            # ====== Deteksi ======
            try:
                res = model.predict(
                    source=infer_src,
                    imgsz=YOLO_IMG,
                    conf=YOLO_CONF,
                    iou=YOLO_IOU,
                    max_det=MAX_DET,
                    device="cpu",
                    verbose=False
                )[0]
            except Exception as e:
                print("[YOLO] inference error:", e)
                time.sleep(0.05)
                continue

            dets  = []
            confs = []

            frame_area = float(H * W)
            min_area   = max(1.0, MIN_BOX_AREA_P * frame_area)

            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls_id = int(b.cls[0]) if hasattr(b, "cls") and b.cls is not None else -1
                    if class_ids is not None and cls_id not in class_ids:
                        continue
                    conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                    # Map kembali bila ROI
                    if roi_px:
                        x1 += roi_px[0]; x2 += roi_px[0]
                        y1 += roi_px[1]; y2 += roi_px[1]
                    # Filter bbox kecil / noise
                    area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                    if area < min_area:
                        continue
                    dets.append((x1, y1, x2, y2, conf, cls_id))
                    confs.append(conf)

            present = len(dets) > 0
            absent_hist.append(0 if present else 1)
            if len(absent_hist) > MISSING_WINDOW:
                absent_hist.pop(0)
            avg_absent = float(np.mean(absent_hist)) if absent_hist else (0.0 if present else 1.0)

            # ====== Keputusan ======
            status    = "NORMAL"
            bar_color = (0, 255, 0)
            if avg_absent >= ALERT_THRESHOLD:
                status    = "ALERT"
                bar_color = (0, 0, 255)
            elif avg_absent >= WARN_THRESHOLD:
                status    = "WARN"
                bar_color = (0, 255, 255)

            # Gambar ROI (opsional)
            if roi_px:
                cv2.rectangle(frame, (roi_px[0], roi_px[1]), (roi_px[2], roi_px[3]), (255, 128, 0), 2)

            # Gambar bbox & label
            for (x1, y1, x2, y2, conf, cls_id) in dets[:MAX_DET]:
                label = model.names.get(cls_id, "obj") if isinstance(model.names, dict) else "obj"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 0), 2, cv2.LINE_AA)

            # Header status & statistik
            if confs:
                cmax = max(confs); cmean = sum(confs) / len(confs)
                conf_info = f" cmax={cmax:.2f} cmean={cmean:.2f}"
            else:
                conf_info = " cmax=-- cmean=--"

            draw_status(
                frame,
                f"Status: {status} | absent≈{avg_absent:.2f} | dets={len(dets)} |{conf_info}",
                color=bar_color
            )

            # Debug: simpan frame saat miss (setiap ~30 frame)
            frame_count += 1
            if not present and SAVE_DEBUG_NOMISS and (frame_count % 30 == 0):
                dbg_dir = os.path.join(BASE_DIR, "debug_nomiss")
                os.makedirs(dbg_dir, exist_ok=True)
                cv2.imwrite(os.path.join(dbg_dir, f"nomiss_{int(time.time())}.jpg"), frame)

            # Notifikasi (cooldown) saat ALERT
            now = time.time()
            if status == "ALERT" and (now - last_alert_ts > TELEGRAM_COOLDOWN):
                last_alert_ts = now
                msg = "⚠️ Kotak infaq TIDAK TERDETEKSI!"
                print("[ALERT]", msg)
                if SAVE_ALERT_FRAME:
                    out_dir = os.path.join(BASE_DIR, "alerts")
                    os.makedirs(out_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(out_dir, f"alert_{int(now)}.jpg"), frame)
                try:
                    asyncio.run(send_telegram(msg))
                    asyncio.run(send_telegram_photo(frame, "⚠️ Kotak infaq hilang dari kamera!"))
                except Exception as e:
                    print("[TG] error:", e)

            if status != last_status:
                print(f"[STATUS] {last_status} -> {status} (absent≈{avg_absent:.2f}, dets={len(dets)})")
                last_status = status

            # ====== Tampil ======
            if SHOW_WINDOW:
                try:
                    cv2.imshow("Kotak Infaq Monitor (press q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as e:
                    # OpenCV build tanpa GUI → jatuh ke headless
                    print("[OpenCV] GUI tidak tersedia, beralih ke headless:", e)
                    os.environ["SHOW_WINDOW"] = "0"
                    break
            else:
                time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[EXIT] keyboard interrupt")
    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
    finally:
        cap.release()
        if SHOW_WINDOW:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
