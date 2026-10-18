"""Convert teammate open_labels.xlsx into data/labels.json."""

from __future__ import annotations

import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
DEFAULT_XLSX = Path(r"C:\Users\User\Downloads\open_labels.xlsx")


def _normalize_stem(stem: str) -> str:
    return stem.translate(str.maketrans("Хх", "Xx")).upper().strip()


def _parse_timestamp(raw: str | None) -> float | None:
    if not raw:
        return None
    text = str(raw).strip().lower().replace("ё", "е")
    if not text or text in {"-", "none", "null"}:
        return None
    minutes = 0
    minute_match = re.search(r"(\d+)\s*мин", text)
    if minute_match:
        minutes = int(minute_match.group(1))
    seconds = None
    sec_match = re.search(r"(\d+(?:[.,]\d+)?)\s*сек", text)
    if sec_match:
        seconds = float(sec_match.group(1).replace(",", "."))
    elif re.fullmatch(r"\d+(?:[.,]\d+)?", text):
        seconds = float(text.replace(",", "."))
    if seconds is None and minutes:
        seconds = 0.0
    if seconds is None:
        return None
    return minutes * 60 + seconds


def _col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def _shared_strings(z: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for si in root.findall("m:si", NS):
        values.append("".join(t.text or "" for t in si.findall(".//m:t", NS)))
    return values


def _cell_text(cell: ET.Element, strings: list[str]) -> str:
    kind = cell.attrib.get("t")
    if kind == "s":
        v = cell.find("m:v", NS)
        if v is not None and v.text is not None:
            return strings[int(v.text)]
    if kind == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//m:t", NS))
    v = cell.find("m:v", NS)
    return v.text if v is not None and v.text else ""


def load_xlsx(path: Path) -> dict[str, dict]:
    with zipfile.ZipFile(path) as z:
        strings = _shared_strings(z)
        sheet_name = next(n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        root = ET.fromstring(z.read(sheet_name))

    videos: dict[str, dict] = {}
    header: list[str] | None = None
    for row in root.findall("m:sheetData/m:row", NS):
        cells: dict[int, str] = {}
        for cell in row.findall("m:c", NS):
            ref = cell.attrib.get("r", "A")
            cells[_col_index(ref)] = _cell_text(cell, strings).strip()
        if not cells:
            continue
        values = [cells.get(i, "") for i in range(max(cells) + 1)]
        if header is None:
            header = [v.lower() for v in values]
            continue
        name = _normalize_stem(Path(values[0]).stem if values else "")
        if not name:
            continue
        timestamp_raw = values[1] if len(values) > 1 else ""
        verdict_raw = (values[2] if len(values) > 2 else "").lower()
        description = values[3] if len(values) > 3 else ""
        code = values[4] if len(values) > 4 else ""
        has_violation = "нарушен" in verdict_raw and "без" not in verdict_raw
        if description.upper() == "NO_VIOLATION":
            has_violation = False
            description = ""
        videos[name] = {
            "has_violation": has_violation,
            "timestamp_sec": _parse_timestamp(timestamp_raw),
            "code": code or None,
            "description": description or None,
        }
    _ = header
    return videos


def main() -> None:
    xlsx = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX
    if not xlsx.exists():
        raise SystemExit(f"xlsx not found: {xlsx}")
    videos = load_xlsx(xlsx)
    payload = {"source": str(xlsx), "videos": videos}
    settings.labels_path.parent.mkdir(parents=True, exist_ok=True)
    settings.labels_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_pos = sum(1 for v in videos.values() if v["has_violation"])
    print(f"Wrote {len(videos)} videos ({n_pos} with violation) -> {settings.labels_path}")


if __name__ == "__main__":
    main()
