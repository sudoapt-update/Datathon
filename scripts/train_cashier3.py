"""Fine-tune the cash-desk detector on the third Roboflow export."""

from collections import Counter
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
names = [
    "bag",
    "cash-in-transit bag",
    "cash_stack",
    "counter",
    "document",
    "door_open",
    "folder",
    "jacket",
    "key_in",
    "phone",
    "safe_open",
    "screen_locked",
    "screen_on",
    "uv_lamp",
]
counts = Counter()
for split in ("train", "valid"):
    files = list((ROOT / "data" / "external_v3" / split / "labels").glob("*.txt"))
    print(split, "labels", len(files), "images", len(list((ROOT / "data" / "external_v3" / split / "images").glob("*"))))
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            counts[names[int(line.split()[0])]] += 1
print("boxes", dict(counts))

model = YOLO(str(ROOT / "yolo11n.pt"))
model.train(
    data=str(ROOT / "data" / "external_v3" / "data.yaml"),
    imgsz=640,
    batch=8,
    workers=0,
    patience=8,
    epochs=40,
    device=0,
    project=str(ROOT / "runs" / "detect"),
    name="cashier3",
    exist_ok=True,
)
