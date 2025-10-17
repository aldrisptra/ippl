import os
from ultralytics import YOLO

weights = os.environ.get("YOLO_WEIGHTS")
if not weights:
    raise SystemExit("Env YOLO_WEIGHTS belum diset.")

m = YOLO(weights)
res = m.predict(
    source="test_cam.jpg",
    conf=0.10,   # sensitif untuk tes awal
    iou=0.45,
    imgsz=640,
    device="cpu",
    save=True,
    verbose=True
)
print("[DONE] Hasil di folder runs/detect/predict*")
