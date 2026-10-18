"""Download CC-BY cash-counter datasets from Roboflow Universe and merge them.

Requires a free key from https://app.roboflow.com/settings/api (env ROBOFLOW_API_KEY).
Gun boxes from Cash Counter Theft are dropped. The merged yaml is
data/external/cashier.yaml — then run training/train_detector.py.

Datasets (workspace, project, version):
  skus-mxdsi/pso_poly          hands, cash, cashier, drawer
  pso-33ka9/pso                same family
  neehal-lingayat-7mktd/currency-in-hand
  jojo-oizxe/hands-in-transaction
  ayaan-falfl/cash-counter-theft   (weapon class removed)
  myominhtet/erke-junction-maw-tin-dataset
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "external"
SOURCES = [
    ("skus-mxdsi", "pso_poly", 1),
    ("pso-33ka9", "pso", 1),
    ("neehal-lingayat-7mktd", "currency-in-hand", 1),
    ("jojo-oizxe", "hands-in-transaction", 1),
    ("ayaan-falfl", "cash-counter-theft", 1),
    ("myominhtet", "erke-junction-maw-tin-dataset", 1),
]
# Names we keep. Everything else, including weapons, is deleted.
KEEP = {
    "person": "person",
    "cashier": "person",
    "customer": "person",
    "hand": "person",
    "laptop": "laptop",
    "phone": "cell phone",
    "cell phone": "cell phone",
    "mobile": "cell phone",
    "handbag": "handbag",
    "bag": "handbag",
    "backpack": "backpack",
    "cash": "cash",
    "currency": "cash",
    "banknote": "cash",
    "money": "cash",
}


def main() -> None:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY to a Roboflow private key, then rerun.")
    from roboflow import Roboflow

    rf = Roboflow(api_key=key)
    merged = OUT / "merged"
    if merged.exists():
        shutil.rmtree(merged)
    for split in ("train", "valid"):
        (merged / "images" / split).mkdir(parents=True)
        (merged / "labels" / split).mkdir(parents=True)
    names = ["person", "laptop", "cell phone", "handbag", "backpack", "cash"]
    index = {name: i for i, name in enumerate(names)}
    for workspace, project, version in SOURCES:
        print("download", workspace, project, version)
        dataset = (
            rf.workspace(workspace)
            .project(project)
            .version(version)
            .download("yolov11", location=str(OUT / "raw" / project))
        )
        _ingest(Path(dataset.location), merged, index)
    yaml_path = OUT / "cashier.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {merged.as_posix()}",
                "train: images/train",
                "val: images/valid",
                "names:",
                *[f"  {i}: {name}" for i, name in enumerate(names)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("wrote", yaml_path)


def _ingest(src: Path, merged: Path, index: dict[str, int]) -> None:
    data_yaml = src / "data.yaml"
    src_names = _read_names(data_yaml)
    for split in ("train", "valid", "test"):
        images = src / split / "images"
        labels = src / split / "labels"
        if not images.exists():
            images = src / "images" / split
            labels = src / "labels" / split
        if not images.exists():
            continue
        dest_split = "valid" if split == "test" else split
        for image in images.iterdir():
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label = labels / f"{image.stem}.txt"
            rewritten = _rewrite_label(label, src_names, index)
            if rewritten is None:
                continue
            dest_image = merged / "images" / dest_split / f"{src.name}_{image.name}"
            dest_label = merged / "labels" / dest_split / f"{src.name}_{image.stem}.txt"
            shutil.copy2(image, dest_image)
            dest_label.write_text(rewritten, encoding="utf-8")


def _read_names(data_yaml: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    if not data_yaml.exists():
        return names
    for line in data_yaml.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key.isdigit() and value:
            names[int(key)] = value.lower()
    return names


def _rewrite_label(path: Path, src_names: dict[int, str], index: dict[str, int]) -> str | None:
    if not path.exists():
        return ""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        raw = src_names.get(cls, "")
        if "gun" in raw or "weapon" in raw or "knife" in raw:
            continue
        mapped = KEEP.get(raw)
        if mapped is None:
            continue
        rows.append(" ".join([str(index[mapped]), *parts[1:]]))
    if not rows:
        return None
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()
