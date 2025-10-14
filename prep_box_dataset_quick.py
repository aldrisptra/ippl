import os
import fiftyone as fo
import fiftyone.zoo as foz

# ===== KONFIGURASI =====
OUTPUT_DIR = r"C:\Users\ALDRI\Documents\PROYEK\dataset_box"  # ganti kalau mau
CLASS_NAME = "Box"        # Open Images class (case-sensitive)
MAX_SAMPLES = 800         # total sample diambil dari TRAIN
VAL_FRACTION = 0.15       # 15% jadi validation

os.makedirs(OUTPUT_DIR, exist_ok=True)
fo.config.database_validation = False  # hilangkan warning MongoDB

print("Mengunduh TRAIN (kelas 'Box')…")
ds = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=[CLASS_NAME],
    max_samples=MAX_SAMPLES,
    only_matching=True,
    seed=51,
)

print("Remap label ke 'box' (single-class)…")
def remap(sample):
    det = sample.ground_truth
    if det is not None:
        for obj in det.detections:
            obj.label = "box"
        sample.ground_truth = det
    sample.save()
ds.map(remap)

print("Split lokal train/val…")
train_view, val_view = ds.random_split([1-VAL_FRACTION, VAL_FRACTION], seed=42)

exporter = dict(
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    export_media="copy",
    classes=["box"],
)

print("Export TRAIN → YOLO…")
train_view.export(export_dir=OUTPUT_DIR, split="train", **exporter)

print("Export VAL → YOLO…")
val_view.export(export_dir=OUTPUT_DIR, split="val", **exporter)

print("DONE →", OUTPUT_DIR)
