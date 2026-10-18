"""Fine-tune YOLO11n on a merged cashier dataset.

Expects a YOLO dataset yaml at data/external/cashier.yaml with images that
already contain the classes the rules consume:

  person, laptop, cell phone, handbag, backpack

Open cash-counter sets (PSO, Currency in hand, Cash Counter Theft, Erke)
can be merged into that yaml first. Gun boxes must be deleted before training.
The checkpoint is written to weights/cashier.pt and the service loads it
automatically on the next start.

Example:
  .\\venv\\Scripts\\python.exe training/train_detector.py --data data/external/cashier.yaml --epochs 20
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "external" / "cashier.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--base", type=str, default="yolo11n.pt")
    args = parser.parse_args()
    if not args.data.exists():
        raise SystemExit(
            f"Dataset yaml not found: {args.data}\n"
            "Put a merged YOLO dataset there (see training/README in the script docstring). "
            "Roboflow exports need a free API key in ROBOFLOW_API_KEY."
        )
    from ultralytics import YOLO

    model = YOLO(args.base)
    run = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=0,
        project=str(ROOT / "runs" / "detect"),
        name="cashier",
        exist_ok=True,
        patience=8,
    )
    best = Path(run.save_dir) / "weights" / "best.pt"
    dest = ROOT / "weights" / "cashier.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest)
    print("saved", dest)


if __name__ == "__main__":
    main()
