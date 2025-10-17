# run_cam.py  — YOLO cam + ROI + debounce + Telegram
import os, time, json, argparse
import numpy as np
import cv2
from ultralytics import YOLO
import httpx

# =========================
# Konfigurasi lewat ENV
# =========================
MISSING_WINDOW   = int(os.getenv("MISSING_WINDOW",  "24"))     # panjang moving avg (frame)
WARN_THRESHOLD   = float(os.getenv("WARN_THRESHOLD",  "0.40")) # rata2 'absent' utk WARN
ALERT_THRESHOLD  = float(os.getenv("ALERT_THRESHOLD", "0.70")) # rata2 'absent' utk ALERT
PRESENT_GRACE    = int(os.getenv("PRESENT_GRACE", "10"))       # toleransi frame hilang

YOLO_WEIGHTS     = os.getenv("YOLO_WEIGHTS", os.path.join(os.getcwd(), "best.pt"))
YOLO_CONF_DEF    = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_IOU_DEF     = float(os.getenv("YOLO_IOU", "0.60"))
YOLO_IMG_DEF     = int(os.getenv("YOLO_IMG",  "800"))
MIN_AREA_RATIO   = float(os.getenv("MIN_AREA_RATIO", "0.01"))  # relative ke area ROI jika ROI aktif

# Telegram (opsional)
TG_TOKEN         = os.getenv("tg_token", "7697921487:AAEvZXLkC61Nzx-eh1e2BES1VfqSJ3wN32E")
TG_CHAT_ID       = os.getenv("tg_chat_id", "1215968232")
TG_COOLDOWN_S    = int(os.getenv("tg_cooldown", "10"))

ROI_JSON         = os.getenv("ROI_JSON", "roi.json")           # lokasi file ROI

# =========================
# Utils
# =========================
def send_telegram(text: str):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        httpx.post(url, data={"chat_id": TG_CHAT_ID, "text": text}, timeout=10.0)
    except Exception as e:
        print("[TG] gagal kirim pesan:", e)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def load_roi(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and all(k in obj for k in ("x1","y1","x2","y2")):
            return obj
    except Exception:
        pass
    return None

def save_roi(path: str, rect: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rect, f)
        print(f"[ROI] disimpan ke {path}: {rect}")
    except Exception as e:
        print("[ROI] gagal simpan:", e)

# =========================
# ROI Interaktif (drag mouse)
# =========================
class ROISelector:
    def __init__(self, win_name="Kotak Infaq Monitor (press H for help)"):
        self.win = win_name
        self.dragging = False
        self.start = None
        self.current = None
        self.rect = None  # dict: {x1,y1,x2,y2}

    def set_rect(self, rect):
        self.rect = rect

    def attach(self):
        cv2.setMouseCallback(self.win, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            if self.start is not None and self.current is not None:
                x1, y1 = self.start
                x2, y2 = self.current
                if abs(x2-x1) > 10 and abs(y2-y1) > 10:
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    self.rect = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            self.start = None
            self.current = None

    def draw_overlay(self, frame):
        # draw saved rect
        if self.rect:
            x1,y1,x2,y2 = self.rect.values()
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 165, 0), 2)  # orange
        # draw dragging rect
        if self.dragging and self.start and self.current:
            x1,y1 = self.start; x2,y2 = self.current
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 255), 1)

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Index kamera (default 0)")
    ap.add_argument("--weights", type=str, default=YOLO_WEIGHTS, help="Path weights YOLO")
    ap.add_argument("--conf", type=float, default=YOLO_CONF_DEF, help="Confidence min deteksi")
    ap.add_argument("--iou", type=float, default=YOLO_IOU_DEF, help="IOU untuk NMS")
    ap.add_argument("--imgsz", type=int, default=YOLO_IMG_DEF, help="Ukuran inferensi YOLO")
    ap.add_argument("--min-area", type=float, default=MIN_AREA_RATIO, help="Minimal area bbox / area ROI")
    ap.add_argument("--roi", type=str, default="", help="Preset ROI x1,y1,x2,y2 (override file)")
    ap.add_argument("--no-tg", action="store_true", help="Matikan notifikasi Telegram")
    args = ap.parse_args()

    print(f"[INFO] Load model: {args.weights}")
    try:
        model = YOLO(args.weights)
    except Exception as e:
        print("[ERROR] Gagal load YOLO:", e)
        return

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak bisa dibuka. Coba ganti --camera 1/2 dst.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ROI state
    roi = None
    if args.roi:
        try:
            x1,y1,x2,y2 = map(int, args.roi.split(","))
            roi = {"x1":x1,"y1":y1,"x2":x2,"y2":y2}
            print("[ROI] preset dari argumen:", roi)
        except Exception:
            print("[ROI] format --roi salah, gunakan x1,y1,x2,y2 (contoh 100,200,800,650)")
    if roi is None:
        roi = load_roi(ROI_JSON)
        if roi: print("[ROI] dimuat dari file:", ROI_JSON, roi)

    win = "Kotak Infaq Monitor (press H for help)"
    cv2.namedWindow(win)
    roi_sel = ROISelector(win)
    roi_sel.set_rect(roi)
    roi_sel.attach()

    # Status
    absent_hist = []
    missing_streak = 0
    last_t = time.time()
    fps = 0.0
    last_alert_ts = 0.0
    last_status = "NORMAL"
    help_on = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Gagal ambil frame dari kamera")
            break

        H, W = frame.shape[:2]

        # Tentukan ROI yang aktif
        active_roi = roi_sel.rect
        if active_roi:
            x1 = clamp(active_roi["x1"], 0, W-1)
            y1 = clamp(active_roi["y1"], 0, H-1)
            x2 = clamp(active_roi["x2"], 1, W)
            y2 = clamp(active_roi["y2"], 1, H)
            if x2 - x1 < 4 or y2 - y1 < 4:
                active_roi = None
        # Deteksi pada crop ROI (jika ada), lalu offset balik
        dets = []
        if active_roi:
            rx1, ry1, rx2, ry2 = active_roi["x1"], active_roi["y1"], active_roi["x2"], active_roi["y2"]
            crop = frame[ry1:ry2, rx1:rx2]
            base_area = max(1, (ry2-ry1)*(rx2-rx1))
            res = model.predict(source=crop, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    conf = float(b.conf[0])
                    x1_, y1_, x2_, y2_ = [int(v) for v in b.xyxy[0]]
                    # offset balik ke koordinat global
                    x1g, y1g, x2g, yg2 = x1_ + rx1, y1_ + ry1, x2_ + rx1, y2_ + ry1
                    area = max(1, (x2g - x1g) * (yg2 - y1g))
                    if area / base_area >= args.min_area:
                        dets.append((x1g, y1g, x2g, yg2, conf))
        else:
            base_area = max(1, H * W)
            res = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                    area = max(1, (x2 - x1) * (y2 - y1))
                    if area / base_area >= args.min_area:
                        dets.append((x1, y1, x2, y2, conf))

        # Debounce temporal
        present_raw = len(dets) > 0
        if present_raw:
            missing_streak = 0
        else:
            missing_streak += 1
        present = True if present_raw or missing_streak <= PRESENT_GRACE else False

        # Moving average
        absent_hist.append(0 if present else 1)
        if len(absent_hist) > MISSING_WINDOW:
            absent_hist.pop(0)
        avg_absent = float(np.mean(absent_hist)) if absent_hist else (0.0 if present else 1.0)

        # Status
        status = "NORMAL"
        if avg_absent >= ALERT_THRESHOLD:
            status = "ALERT"
        elif avg_absent >= WARN_THRESHOLD:
            status = "WARN"

        # Telegram saat naik ke ALERT
        if (not args.no_tg) and TG_TOKEN and TG_CHAT_ID and status == "ALERT":
            now = time.time()
            if now - last_alert_ts > TG_COOLDOWN_S and last_status != "ALERT":
                last_alert_ts = now
                send_telegram("⚠️ Kotak infaq tidak terdeteksi di kamera! (status=ALERT)")
        last_status = status

        # ===== Draw =====
        # ROI overlay
        roi_sel.draw_overlay(frame)

        # bbox
        for (x1, y1, x2, y2, conf) in dets[:10]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, f"kotakinfaq {conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)

        # FPS
        now = time.time()
        dt = now - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_t = now

        # HUD
        hud = f"Status: {status} | absent≈{avg_absent:.2f} | dets={len(dets)} | FPS={fps:.1f}"
        color = (0, 255, 255) if status == "NORMAL" else ((0, 165, 255) if status == "WARN" else (0, 0, 255))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        if help_on:
            help_lines = [
                "[H] toggle help   [R] draw/ubah ROI (drag kiri mouse)",
                "[S] simpan ROI ke roi.json   [C] clear ROI",
                "[Q]/[ESC] quit",
            ]
            y = 60
            for ln in help_lines:
                cv2.putText(frame, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                y += 24

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('h'):
            help_on = not help_on
        elif key == ord('r'):
            # hanya memastikan callback aktif; user tinggal drag
            print("[ROI] Drag kiri mouse pada area jendela untuk menentukan ROI, lalu tekan [S] untuk simpan.")
        elif key == ord('s'):
            if roi_sel.rect:
                save_roi(ROI_JSON, roi_sel.rect)
            else:
                print("[ROI] belum ada ROI untuk disimpan (drag dulu dengan [R]).")
        elif key == ord('c'):
            roi_sel.rect = None
            print("[ROI] cleared.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
