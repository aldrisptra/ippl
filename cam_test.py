import cv2, time

BACKENDS = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
INDICES  = [0,1,2,3,4]

print("OpenCV:", cv2.__version__)
ok_any = False

for name, cap_flag in BACKENDS:
    print(f"\n=== Test backend: {name} ===")
    for i in INDICES:
        cap = cv2.VideoCapture(i, cap_flag)
        ok = cap.isOpened()
        print(f"  Index {i}: {'OPENED' if ok else '----'}")
        if ok:
            # coba baca 1 frame
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"    -> READ OK, frame: {w}x{h}")
                    ok_any = True
                    break
                time.sleep(0.05)
            cap.release()

if not ok_any:
    print("\nTidak ada kombinasi backend/index yang berhasil buka & baca frame.")
    print("Coba tutup aplikasi lain yg pakai kamera, cek izin Windows, atau restart.")
