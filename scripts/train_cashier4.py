"""Fine-tune the cash-desk detector on the next Roboflow export.

The export must be the whole project, not only the new ID and banknote photos.
Class names stay the 14 names from v3. Production weights are left in place;
this script writes runs/detect/cashier4 and does not replace weights/cashier.pt.
"""

from collections import Counter
from pathlib import Path
import sys

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "external_v4"
NAMES = [
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
MIN_TRAIN_IMAGES = 278


def main() -> None:
    yaml_path = DATA / "data.yaml"
    if not yaml_path.exists():
        sys.exit(f"No export at {DATA}. Put the YOLOv11 zip contents there first.")
    text = yaml_path.read_text(encoding="utf-8")
    if "names:" not in text:
        sys.exit("data.yaml has no class names.")
    for name in NAMES:
        if name not in text:
            sys.exit(f"data.yaml is missing class {name!r}. Do not rename or drop classes.")

    counts: Counter[str] = Counter()
    for split in ("train", "valid"):
        label_dir = DATA / split / "labels"
        image_dir = DATA / split / "images"
        labels = list(label_dir.glob("*.txt"))
        images = [path for path in image_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        print(split, "labels", len(labels), "images", len(images))
        if split == "train" and len(images) < MIN_TRAIN_IMAGES:
            sys.exit(
                f"Train split has {len(images)} images. v3 had {MIN_TRAIN_IMAGES}. "
                "Export the full project, including the old cash-desk frames."
            )
        for path in labels:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                counts[NAMES[int(line.split()[0])]] += 1
    print("boxes", dict(counts))
    print("document", counts["document"], "cash_stack", counts["cash_stack"])

    model = YOLO(str(ROOT / "weights" / "cashier.pt"))
    model.train(
        data=str(yaml_path),
        imgsz=640,
        batch=8,
        workers=0,
        patience=8,
        epochs=30,
        device=0,
        project=str(ROOT / "runs" / "detect"),
        name="cashier4",
        exist_ok=True,
    )
    print("Done. Weights are in runs/detect/cashier4/weights/best.pt")
    print("weights/cashier.pt was not replaced.")


if __name__ == "__main__":
    main()
