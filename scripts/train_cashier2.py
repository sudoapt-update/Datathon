"""Fine-tune the cash-desk detector on the second Roboflow export."""

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="data/external_v2/data.yaml",
    imgsz=640,
    batch=8,
    workers=0,
    patience=8,
    epochs=40,
    device=0,
    project="runs/detect",
    name="cashier2",
    exist_ok=True,
)
