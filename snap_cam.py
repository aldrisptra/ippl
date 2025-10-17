# snap_cam.py
import os, cv2

IDX = int(os.environ.get("CAM_INDEX", "0"))

# Coba DSHOW dulu (umum di Windows), bila gagal fallback ke MSMF lalu ANY
for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
    cap = cv2.VideoCapture(IDX, backend)
    if cap.isOpened():
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            out = "test_cam.jpg"
            cv2.imwrite(out, frame)
            print(f"[OK] Tersimpan -> {out}, shape={frame.shape}, backend={backend}")
            break
    else:
        cap.release()
else:
    print("[ERR] Tidak bisa buka kamera. Coba set CAM_INDEX=1 atau 2, atau pastikan kamera tidak dipakai app lain.")
