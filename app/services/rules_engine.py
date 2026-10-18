"""Temporal rules over a person/object detector.

YOLO11m finds people, phones, bags and the laptop that anchors the desk.
weights/cashier.pt, when present, adds the labeled cash-desk classes on top.
Each catalog id has its own rule and emits the second where the evidence is
strongest. Legal wording comes only from app/rules/catalog.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from app.config import ROOT_DIR, settings
from app.schemas import EvidenceFrame, Violation
from app.services.video import evidence_frames, iter_bgr_frames

_COCO = None
_CASHIER = None
_POSE = None

BAG_NAMES = {"handbag", "backpack", "suitcase"}
KEEP_NAMES = BAG_NAMES | {"person", "cell phone", "laptop"}
CASHIER_NAMES = {
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
}


@dataclass
class Det:
    name: str
    conf: float
    cx: float
    cy: float
    x1: float
    y1: float
    x2: float
    y2: float
    area: float


@dataclass
class Sample:
    t: float
    dets: list[Det] = field(default_factory=list)
    wrists: list[tuple[float, float]] = field(default_factory=list)
    screen_brightness: float | None = None
    desk_edges: float | None = None


@dataclass
class Hit:
    rule_id: str
    t: float
    observation: str
    evidence_at: list[float] | None = None


def load_zones() -> dict:
    path = ROOT_DIR / "app" / "rules" / "zones.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _to_cuda(model):
    try:
        import torch

        if torch.cuda.is_available():
            model.to("cuda")
    except Exception:
        pass
    return model


def load_detector():
    """Load COCO weights and, when trained, the cash-desk checkpoint."""
    global _COCO, _CASHIER
    if _COCO is None:
        from ultralytics import YOLO

        _COCO = _to_cuda(YOLO(str(settings.yolo_weights)))
    trained = ROOT_DIR / "weights" / "cashier.pt"
    if _CASHIER is None and trained.exists():
        from ultralytics import YOLO

        _CASHIER = _to_cuda(YOLO(str(trained)))
    return _COCO


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _parse(result, allowed: set[str]) -> list[Det]:
    height, width = result.orig_shape
    names = result.names
    out: list[Det] = []
    if result.boxes is None:
        return out
    for box in result.boxes:
        name = names[int(box.cls[0])]
        if name not in allowed:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append(
            Det(
                name=name,
                conf=float(box.conf[0]),
                cx=(x1 + x2) / 2 / width,
                cy=(y1 + y2) / 2 / height,
                x1=x1 / width,
                y1=y1 / height,
                x2=x2 / width,
                y2=y2 / height,
                area=(x2 - x1) * (y2 - y1) / (width * height),
            )
        )
    return out


def load_pose():
    global _POSE
    if _POSE is None:
        from ultralytics import YOLO

        _POSE = _to_cuda(YOLO(str(ROOT_DIR / "yolo11n-pose.pt")))
    return _POSE


def _wrists_of(result) -> list[tuple[float, float]]:
    if result.keypoints is None or result.keypoints.xy is None:
        return []
    height, width = result.orig_shape
    points: list[tuple[float, float]] = []
    for person in result.keypoints.xy:
        for index in (9, 10):
            x, y = person[index].tolist()
            if x <= 0 and y <= 0:
                continue
            points.append((x / width, y / height))
    return points


def collect_samples(video_path: Path, fps: float) -> list[Sample]:
    coco = load_detector()
    pose = load_pose()
    samples: list[Sample] = []
    for timestamp, bgr in iter_bgr_frames(video_path, fps=fps):
        dets = _parse(coco(bgr, verbose=False, conf=0.25)[0], KEEP_NAMES)
        if _CASHIER is not None:
            dets.extend(_parse(_CASHIER(bgr, verbose=False, conf=0.25)[0], CASHIER_NAMES))
        wrists = _wrists_of(pose(bgr, verbose=False, conf=0.25)[0])
        brightness, edges = _desk_surface(bgr, dets)
        samples.append(
            Sample(
                t=float(timestamp),
                dets=dets,
                wrists=wrists,
                screen_brightness=brightness,
                desk_edges=edges,
            )
        )
    return samples


def _desk_surface(bgr, dets: list[Det]) -> tuple[float | None, float | None]:
    """Brightness of the laptop screen, and how much is lying on the desk between it and the counter."""
    laptops = [det for det in dets if det.name == "laptop" and det.conf >= 0.45]
    if not laptops:
        return None, None
    height, width = bgr.shape[:2]
    laptop = max(laptops, key=lambda det: det.conf)
    x1 = int(laptop.x1 * width)
    y1 = int(laptop.y1 * height)
    x2 = int(laptop.x2 * width)
    y2 = int(laptop.y2 * height)
    pad_x = max(1, (x2 - x1) // 5)
    pad_y = max(1, (y2 - y1) // 5)
    screen = bgr[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
    brightness = float(screen.mean()) if screen.size else None
    counters = [det for det in dets if det.name == "counter" and det.conf >= 0.5]
    if not counters:
        return brightness, None
    counter = max(counters, key=lambda det: det.conf)
    left = int(min(counter.x2, laptop.x1) * width)
    right = int(max(counter.x2, laptop.x1) * width)
    cy = int(((counter.cy + laptop.cy) / 2) * height)
    top = max(0, cy - int(0.04 * height))
    bottom = cy + int(0.05 * height)
    if right - left < 8:
        return brightness, None
    desk = bgr[top:bottom, max(0, left) : right]
    if desk.size == 0:
        return brightness, None
    gray = cv2.cvtColor(desk, cv2.COLOR_BGR2GRAY)
    edges = float(cv2.Canny(gray, 40, 120).mean())
    return brightness, edges


def _people(sample: Sample, zones: dict, min_cx: float | None = None) -> list[Det]:
    cx_min = zones["person_min_cx"] if min_cx is None else min_cx
    return [
        d
        for d in sample.dets
        if d.name == "person" and d.area >= zones["person_min_area"] and d.cx >= cx_min
    ]


def _median_point(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    xs = sorted(p[0] for p in points)
    ys = sorted(p[1] for p in points)
    mid = len(points) // 2
    return xs[mid], ys[mid]


def _cashier_anchor(samples: list[Sample], zones: dict) -> tuple[float, float] | None:
    """Person closest to the camera during the opening seconds — the cashier."""
    points: list[tuple[float, float]] = []
    for sample in samples:
        if sample.t > zones["anchor_head_sec"]:
            break
        people = _people(sample, zones)
        if not people:
            continue
        cashier = max(people, key=lambda d: d.cy)
        points.append((cashier.cx, cashier.cy))
    return _median_point(points)


def _laptop_point(samples: list[Sample]) -> tuple[float, float] | None:
    points = []
    for sample in samples:
        laptops = [d for d in sample.dets if d.name == "laptop" and d.conf >= 0.45]
        if not laptops:
            continue
        best = max(laptops, key=lambda d: d.conf)
        points.append((best.cx, best.cy))
    return _median_point(points)


def _near(sample: Sample, zones: dict, anchor: tuple[float, float], radius: float) -> bool:
    return any(_dist(p.cx, p.cy, anchor[0], anchor[1]) <= radius for p in _people(sample, zones))


def _dt(samples: list[Sample], index: int) -> float:
    if index == 0:
        return 0.0
    return max(0.0, samples[index].t - samples[index - 1].t)


def decide(samples: list[Sample], zones: dict | None = None) -> list[Hit]:
    zones = zones or load_zones()
    if len(samples) < 3:
        return []
    anchor = _cashier_anchor(samples, zones)
    laptop = _laptop_point(samples)
    hits: list[Hit] = []
    if anchor is not None:
        leave = _rule_leave(samples, zones, anchor)
        if leave:
            hits.append(leave)
    if laptop is not None:
        bag = _rule_bag(samples, zones, laptop)
        if bag:
            hits.append(bag)
        phone = _rule_phone(samples, zones, laptop)
        if phone:
            hits.append(phone)
    crowd = _rule_crowd(samples, zones)
    if crowd:
        hits.append(crowd)
    hits.extend(_rule_desk_objects(samples, zones, anchor))
    return _dedupe(hits)


def _seen(sample: Sample, name: str, conf: float) -> bool:
    return any(det.name == name and det.conf >= conf for det in sample.dets)


def _full_lock(sample: Sample, zones: dict) -> bool:
    """A lock screen that covers the laptop. A small box on a working display does not count."""
    return any(
        det.name == "screen_locked"
        and det.conf >= zones["leave_lock_conf"]
        and det.area >= zones["leave_lock_area"]
        for det in sample.dets
    )


def _desk_in_service(sample: Sample, history: list[Sample], zones: dict) -> bool:
    """The workplace is still open. A locked or dark empty desk may be left."""
    if _seen(sample, "screen_on", 0.45):
        return True
    if any(det.conf >= 0.5 for det in _desk_cash(sample, zones)):
        return True
    if sample.desk_edges is not None and sample.desk_edges >= zones["leave_desk_edges"]:
        return True
    if _full_lock(sample, zones):
        return False
    if sample.screen_brightness is not None and sample.screen_brightness < zones["leave_dark_screen"]:
        return False
    laptop = _seen(sample, "laptop", 0.45)
    locked = _seen(sample, "screen_locked", 0.45) and not _seen(sample, "screen_on", 0.4)
    if locked and not laptop:
        return False
    if laptop:
        return True
    had_screen = any(_seen(prev, "screen_on", 0.45) for prev in history)
    return not had_screen and laptop


def _stepped_away(sample: Sample, zones: dict, anchor: tuple[float, float]) -> bool:
    """Toward the door or out of the frame. Standing beside the desk does not count."""
    people = _people(sample, zones, min_cx=0.0)
    if not people:
        return True
    person = min(people, key=lambda det: _dist(det.cx, det.cy, anchor[0], anchor[1]))
    if person.cy < anchor[1] - zones["leave_door_dy"]:
        return True
    return person.cx < anchor[0] - zones["leave_door_dx"]


def _desk_still_open_phrase(sample: Sample, history: list[Sample], zones: dict) -> str | None:
    """What on this frame shows the workplace was not shut down. None if it looks closed."""
    if not _desk_in_service(sample, history, zones):
        return None
    if _seen(sample, "screen_on", 0.45):
        return "экран на столе всё ещё включён"
    if any(det.conf >= 0.5 for det in _desk_cash(sample, zones)):
        return "на столе лежат деньги"
    if _seen(sample, "laptop", 0.45):
        return "ноутбук на столе не выключен"
    if sample.desk_edges is not None and sample.desk_edges >= zones["leave_desk_edges"]:
        return "на столе остались документы"
    return "рабочее место не закрыто"


def _cashier_where(sample: Sample, zones: dict, anchor: tuple[float, float]) -> str:
    people = _people(sample, zones, min_cx=0.0)
    if not people:
        return "кассира в кадре уже нет"
    if _stepped_away(sample, zones, anchor):
        return "кассир уже у двери, не за столом"
    return "кассир отошла от стола"


def _rule_leave(samples: list[Sample], zones: dict, anchor: tuple[float, float]) -> Hit | None:
    radius = zones["cashier_radius"]
    present_need = zones["leave_present_sec"]
    away_need = zones["leave_away_sec"]
    present_sec = 0.0
    away_sec = 0.0
    away_start: float | None = None
    armed = False
    episode: list[Sample] = []
    fired_at: float | None = None
    fire_index: int | None = None
    saved_episode: list[Sample] = []
    tail: list[Sample] = []
    back_sec = 0.0

    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        here = _near(sample, zones, anchor, radius)
        if fired_at is not None and fire_index is not None:
            if here:
                back_sec += step
                if back_sec >= 1.0:
                    break
            else:
                back_sec = 0.0
                if sample.t > samples[fire_index].t:
                    tail.append(sample)
                if sample.t - samples[fire_index].t >= 8.0:
                    break
            continue
        if here:
            present_sec += step
            if present_sec >= present_need:
                armed = True
            away_sec = 0.0
            away_start = None
            episode = []
            continue
        if not armed:
            continue
        if away_start is None:
            away_start = sample.t
        away_sec += step
        episode.append(sample)
        if away_sec < away_need or away_start is None:
            continue
        history = samples[: index + 1]
        stepped = [frame for frame in episode if _stepped_away(frame, zones, anchor)]
        if not stepped:
            continue
        open_frames = sum(1 for frame in stepped if _desk_in_service(frame, history, zones))
        if open_frames >= max(1, int(0.6 * len(stepped))):
            fired_at = away_start
            fire_index = index
            saved_episode = list(episode)
    if fired_at is None:
        return None

    before = [frame for frame in samples if fired_at - 4.0 <= frame.t < fired_at]
    at_desk = [
        frame
        for frame in before
        if _near(frame, zones, anchor, radius) and _desk_still_open_phrase(frame, samples, zones)
    ]
    window = saved_episode + tail
    open_away = [
        frame
        for frame in window
        if _stepped_away(frame, zones, anchor) and _desk_still_open_phrase(frame, samples, zones)
    ]
    desk_shot = at_desk[-1] if at_desk else None
    leaving = next((frame for frame in open_away if desk_shot is None or frame.t >= desk_shot.t + 1.5), None)
    if leaving is None and open_away:
        leaving = open_away[0]
    gone_pool = [
        frame
        for frame in open_away
        if leaving is None or frame.t >= leaving.t + 2.0
    ]
    gone = gone_pool[-1] if gone_pool else (open_away[-1] if open_away else None)
    proof_frames = [frame for frame in (desk_shot, leaving, gone) if frame is not None]
    proof: list[float] = []
    for frame in proof_frames:
        if all(abs(frame.t - previous) >= 1.2 for previous in proof):
            proof.append(frame.t)
    shown = gone or leaving or desk_shot
    if shown is None:
        return Hit(
            rule_id="cashier_leaves_during_service",
            t=fired_at,
            observation="Кассир ушла от стола, не закрыв рабочее место.",
        )
    open_phrase = _desk_still_open_phrase(shown, samples, zones) or "рабочее место не закрыто"
    where = _cashier_where(shown, zones, anchor)
    minutes = int(shown.t // 60)
    seconds = shown.t - minutes * 60
    clock = f"{minutes:02d}:{seconds:04.1f}"
    return Hit(
        rule_id="cashier_leaves_during_service",
        t=shown.t,
        observation=f"Кассир уходит от стола, не закрыв кассу. На кадре {clock} {where}, {open_phrase}.",
        evidence_at=proof or None,
    )


def _merge_people(people: list[Det], gap: float) -> list[Det]:
    kept: list[Det] = []
    for person in sorted(people, key=lambda d: d.area, reverse=True):
        if any(_dist(person.cx, person.cy, other.cx, other.cy) < gap for other in kept):
            continue
        kept.append(person)
    return kept


def _drop_glass(people: list[Det], zones: dict) -> list[Det]:
    """A reflection in the left glass is a short box. A real person there is taller."""
    max_cx = zones["crowd_reflect_max_cx"]
    max_h = zones["crowd_reflect_max_h"]
    return [person for person in people if person.cx > max_cx or (person.y2 - person.y1) > max_h]


def _rule_crowd(samples: list[Sample], zones: dict) -> Hit | None:
    need = zones["crowd_persist_sec"]
    count_need = zones["crowd_count"]
    run = 0.0
    start: float | None = None
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        people = _merge_people(
            _drop_glass(_people(sample, zones, min_cx=zones["crowd_min_cx"]), zones),
            zones["crowd_merge_dist"],
        )
        n = len(people)
        if n >= count_need:
            if start is None:
                start = sample.t
            run += step
            if run >= need and start is not None and start >= 1.0:
                return Hit(
                    rule_id="unauthorized_person_in_cash_desk",
                    t=start,
                    observation="В помещении кассы одновременно находятся больше двух человек.",
                )
        else:
            run = 0.0
            start = None
    return None


def _rule_bag(samples: list[Sample], zones: dict, laptop: tuple[float, float]) -> Hit | None:
    need = zones["bag_persist_sec"]
    run = 0.0
    start: float | None = None
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        bags = [
            d
            for d in sample.dets
            if d.name in BAG_NAMES
            and d.conf >= zones["bag_conf"]
            and _dist(d.cx, d.cy, laptop[0], laptop[1]) <= zones["bag_near_laptop"]
            and d.cy >= laptop[1] - zones["bag_above_laptop"]
            and not _inside_person(d, sample)
        ]
        if bags:
            if start is None:
                start = sample.t
            run += step
            if run >= need and start is not None:
                return Hit(
                    rule_id="personal_items_in_cash_desk",
                    t=start,
                    observation="На рабочем столе кассы лежит личная сумка.",
                )
        else:
            run = 0.0
            start = None
    return None


def _inside_person(obj: Det, sample: Sample) -> bool:
    for person in sample.dets:
        if person.name != "person":
            continue
        if person.x1 <= obj.cx <= person.x2 and person.y1 <= obj.cy <= person.y2:
            return True
    return False


def _rule_phone(samples: list[Sample], zones: dict, laptop: tuple[float, float]) -> Hit | None:
    need = zones["phone_persist_sec"]
    run = 0.0
    start: float | None = None
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        phones = [
            d
            for d in sample.dets
            if d.name == "cell phone"
            and d.conf >= zones["phone_conf"]
            and _dist(d.cx, d.cy, laptop[0], laptop[1]) >= zones["phone_away_from_laptop"]
        ]
        if phones:
            if start is None:
                start = sample.t
            run += step
            if run >= need and start is not None:
                return Hit(
                    rule_id="phone_with_camera_in_cash_desk",
                    t=start,
                    observation="В зоне кассы виден смартфон не на месте стационарного телефона.",
                )
        else:
            run = 0.0
            start = None
    return None


def _dedupe(hits: list[Hit]) -> list[Hit]:
    best: dict[str, Hit] = {}
    for hit in hits:
        prev = best.get(hit.rule_id)
        if prev is None or hit.t < prev.t:
            best[hit.rule_id] = hit
    return list(best.values())


def _objs(sample: Sample, name: str, conf: float) -> list[Det]:
    return [d for d in sample.dets if d.name == name and d.conf >= conf]


def _rule_desk_objects(
    samples: list[Sample], zones: dict, anchor: tuple[float, float] | None
) -> list[Hit]:
    """Rules that read the cash-desk classes. One hit per catalog id."""
    hits: list[Hit] = []
    conf = zones["obj_conf"]
    door = _persist_door(samples, zones, conf)
    if door:
        hits.append(door)
    safe = _persist_safe(samples, zones, conf)
    if safe:
        hits.append(safe)
    idle = _persist_cash_idle(samples, zones, anchor, conf)
    if idle:
        hits.append(idle)
    leftover = _cash_when_client_arrives(samples, zones)
    if leftover:
        hits.append(leftover)
    recount = _count_without_client(samples, zones)
    if recount:
        hits.append(recount)
    labeled_phone = _labeled_phone(samples, zones)
    if labeled_phone:
        hits.append(labeled_phone)
    cit = _cit_bag_left(samples, zones, anchor)
    if cit:
        hits.append(cit)
    received = _cit_not_counted(samples, zones)
    if received:
        hits.append(received)
    identity = _cit_without_document(samples, zones)
    if identity:
        hits.append(identity)
    return hits


def _desk_cash(sample: Sample, zones: dict, conf: float | None = None) -> list[Det]:
    stacks = _objs(sample, "cash_stack", zones["cash_desk_conf"] if conf is None else conf)
    counters = _objs(sample, "counter", 0.5)
    kept: list[Det] = []
    for stack in stacks:
        on_machine = any(
            counter.x1 - 0.05 <= stack.cx <= counter.x2 + 0.05
            and counter.y1 - 0.05 <= stack.cy <= counter.y2 + 0.05
            for counter in counters
        )
        if not on_machine:
            kept.append(stack)
    return kept


def _arrival(sample: Sample, zones: dict) -> bool:
    """A standing person is still in the doorway, not yet at the desk."""
    people = _people(sample, zones, min_cx=0.0)
    if len(people) < 2:
        return False
    cashier = max(people, key=lambda person: person.cy)
    for person in people:
        if person is cashier:
            continue
        height = person.y2 - person.y1
        if height < zones["arrival_min_h"] or person.cx > zones["arrival_max_cx"]:
            continue
        if cashier.cy - person.cy >= 0.12:
            return True
    return False


def _cash_when_client_arrives(samples: list[Sample], zones: dict) -> Hit | None:
    """Leftover notes: the stack is already still before this person reaches the doorway."""
    stable = 0.0
    gap = 0.0
    prev: tuple[float, float] | None = None
    held_since: float | None = None
    arrival_since: float | None = None
    was_arriving = False
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        stacks = _desk_cash(sample, zones, conf=0.25)
        if stacks:
            stack = (
                max(stacks, key=lambda det: det.conf)
                if prev is None
                else min(stacks, key=lambda det: _dist(det.cx, det.cy, prev[0], prev[1]))
            )
            jumped = prev is not None and _dist(stack.cx, stack.cy, prev[0], prev[1]) > zones["cash_stable_move"]
            if jumped or held_since is None:
                stable = 0.0
                held_since = sample.t
            else:
                stable += step + gap
            gap = 0.0
            prev = (stack.cx, stack.cy)
        else:
            gap += step
            if gap > 1.6:
                stable = 0.0
                gap = 0.0
                prev = None
                held_since = None
        arriving = _arrival(sample, zones)
        if arriving and not was_arriving:
            arrival_since = sample.t
        if not arriving:
            arrival_since = None
        was_arriving = arriving
        if (
            arriving
            and held_since is not None
            and arrival_since is not None
            and held_since <= arrival_since - 0.8
            and stable >= zones["cash_stable_sec"]
        ):
            return Hit(
                rule_id="cash_left_unsecured",
                t=arrival_since,
                observation="Новый клиент в дверях, а на столе ещё лежат деньги предыдущей операции.",
            )
    return None


def _client_at_window(sample: Sample, zones: dict) -> bool:
    people = _people(sample, zones, min_cx=zones["person_min_cx"])
    if len(people) < 2:
        return False
    cashier = max(people, key=lambda person: person.cy)
    for person in people:
        if person is cashier:
            continue
        if person.cx < zones["client_window_cx"]:
            continue
        if cashier.cy - person.cy >= 0.08:
            return True
    return False


def _wrist_in_counter(sample: Sample) -> bool:
    counters = _objs(sample, "counter", 0.5)
    if not counters or not sample.wrists:
        return False
    counter = max(counters, key=lambda det: det.conf)
    return any(counter.x1 <= x <= counter.x2 and counter.y1 <= y <= counter.y2 for x, y in sample.wrists)


def _hands_near(sample: Sample, box: Det, dist: float) -> bool:
    if not sample.wrists:
        return False
    if any(
        box.x1 - 0.04 <= x <= box.x2 + 0.04 and box.y1 - 0.04 <= y <= box.y2 + 0.04
        for x, y in sample.wrists
    ):
        return True
    return min(_dist(x, y, box.cx, box.cy) for x, y in sample.wrists) <= dist


def _wrist_on_uv(sample: Sample) -> bool:
    lamps = _objs(sample, "uv_lamp", 0.5)
    if not lamps:
        return False
    lamp = max(lamps, key=lambda det: det.conf)
    return _hands_near(sample, lamp, 0.06)


def _cit_bag_left(
    samples: list[Sample], zones: dict, anchor: tuple[float, float] | None
) -> Hit | None:
    if anchor is None:
        return None
    run = 0.0
    start: float | None = None
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        bags = _objs(sample, "cash-in-transit bag", 0.5)
        away = _stepped_away(sample, zones, anchor) or not _near(sample, zones, anchor, zones["cashier_radius"])
        if bags and away:
            if start is None:
                start = sample.t
            run += step
            if run >= zones["cit_away_sec"] and start is not None:
                return Hit(
                    rule_id="cashier_leaves_during_service",
                    t=start,
                    observation="Инкассаторская сумка осталась на столе, а кассир отошла от места.",
                )
        else:
            run = 0.0
            start = None
    return None


def _cit_not_counted(samples: list[Sample], zones: dict) -> Hit | None:
    """Money came out of the collection bag and never went through the counter."""
    bag_run = 0.0
    cash_run = 0.0
    wrist = 0.0
    ready: float | None = None
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        if _wrist_in_counter(sample):
            wrist += step
        if _objs(sample, "cash-in-transit bag", 0.5):
            bag_run += step
        else:
            bag_run = 0.0
        if _objs(sample, "cash_stack", 0.4):
            cash_run += step
        else:
            cash_run = 0.0
        if bag_run >= 2.0 and cash_run >= 2.0 and ready is None:
            ready = sample.t
    if ready is None or wrist >= 0.5:
        return None
    return Hit(
        rule_id="no_authenticity_check",
        t=ready,
        observation="Купюры уже вынуты из инкассаторской сумки, а через счётную машину они не проходили.",
    )


def _document_in_hand(sample: Sample, zones: dict) -> bool:
    """The identity card is in a hand, not lying on the desk."""
    docs = _objs(sample, "document", 0.4)
    if not docs:
        return False
    for doc in docs:
        if sample.wrists and min(_dist(doc.cx, doc.cy, x, y) for x, y in sample.wrists) <= 0.11:
            return True
        for person in _people(sample, zones, min_cx=0.0):
            inside = person.x1 <= doc.cx <= person.x2 and person.y1 <= doc.cy <= person.y2
            if inside and doc.cy <= person.cy:
                return True
    return False


def _id_hold_done(samples: list[Sample], zones: dict, before: float | None = None) -> bool:
    """True when the cashier has held the document for at least 1.5 seconds."""
    need = zones["id_hold_sec"]
    run = 0.0
    gap = 0.0
    for index, sample in enumerate(samples):
        if before is not None and sample.t > before:
            break
        step = _dt(samples, index)
        if _document_in_hand(sample, zones):
            run += step
            gap = 0.0
            if run >= need:
                return True
        else:
            gap += step
            if gap > 0.8:
                run = 0.0
                gap = 0.0
    return False


def _cit_without_document(samples: list[Sample], zones: dict) -> Hit | None:
    """A standing visitor hands over the collection bag before any identity document."""
    if any(_objs(sample, "document", 0.4) for sample in samples):
        return None
    run = 0.0
    start: float | None = None
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        people = _people(sample, zones, min_cx=0.0)
        bag = bool(_objs(sample, "cash-in-transit bag", 0.5))
        cashier = max(people, key=lambda person: person.cy) if people else None
        visitor = cashier is not None and any(
            person is not cashier and (person.y2 - person.y1) >= 0.24 for person in people
        )
        if bag and visitor:
            if start is None:
                start = sample.t
            run += step
            if run >= 1.0 and start is not None:
                return Hit(
                    rule_id="no_client_identification",
                    t=start,
                    observation="Инкассатор передал сумку, а документ кассир не взяла и с лицом не сверила.",
                )
        else:
            run = 0.0
            start = None
    return None


def _remainder_without_recount(samples: list[Sample], zones: dict) -> Hit | None:
    counted = 0.0
    had_count = False
    idle = 0.0
    start: float | None = None
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        if _wrist_in_counter(sample):
            counted += step
            if counted >= zones["count_session_sec"]:
                had_count = True
            idle = 0.0
            start = None
            continue
        counted = 0.0
        cash = _desk_cash(sample, zones)
        if not had_count or not _client_at_window(sample, zones) or not cash:
            idle = 0.0
            start = None
            continue
        if start is None:
            start = sample.t
        idle += step
        if idle >= zones["remainder_sec"] and start is not None:
            return Hit(
                rule_id="cash_in_pocket",
                t=start,
                observation="После выдачи части денег кассир не пересчитала остаток в машинке, а сразу упаковывает его.",
            )
    return None


def _cash_before_id_hold(samples: list[Sample], zones: dict) -> Hit | None:
    """Checking money counts as identified only after the document was held for 1.5 seconds.

    A card later laid on the desk is the normal next step, not a substitute for that hold.
    """
    client_seen = 0.0
    client_ready = False
    count_run = 0.0
    count_start: float | None = None
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        if _client_at_window(sample, zones):
            client_seen += step
            if client_seen >= zones["client_present_sec"]:
                client_ready = True
        if not client_ready or not _wrist_in_counter(sample):
            count_run = 0.0
            count_start = None
            continue
        if count_start is None:
            count_start = sample.t
        count_run += step
        if count_run < zones["count_wrist_sec"] or count_start is None:
            continue
        if _id_hold_done(samples, zones, before=count_start):
            return None
        return Hit(
            rule_id="no_client_identification",
            t=count_start,
            observation=(
                "Кассир начала проверять деньги, не подержав документ клиента в руке хотя бы 1,5 секунды. "
                "Карточка, которую потом кладут на стол, эту проверку не заменяет."
            ),
        )
    return None


def _no_identification(samples: list[Sample], zones: dict) -> Hit | None:
    wait = 0.0
    start: float | None = None
    saw_document = False
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        if _objs(sample, "document", 0.4):
            saw_document = True
        serving = _client_at_window(sample, zones)
        working = _wrist_in_counter(sample) or bool(_desk_cash(sample, zones))
        if not serving:
            wait = 0.0
            start = None
            continue
        if saw_document:
            wait = 0.0
            start = None
            continue
        if not working:
            continue
        if start is None:
            start = sample.t
        wait += step
        if wait >= zones["id_wait_sec"] and start is not None:
            return Hit(
                rule_id="no_client_identification",
                t=start,
                observation="Клиент уже у окна, а кассир начала операцию, не проверив документ.",
            )
    return None


def _merge_without_uv(samples: list[Sample], zones: dict) -> Hit | None:
    counted = 0.0
    after = False
    idle = 0.0
    start: float | None = None
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        if _wrist_in_counter(sample):
            counted += step
            if counted >= zones["count_session_sec"]:
                after = True
            idle = 0.0
            start = None
            continue
        counted = 0.0
        if not after:
            continue
        if _wrist_on_uv(sample) and not _wrist_in_counter(sample):
            after = False
            idle = 0.0
            start = None
            continue
        stacks = _desk_cash(sample, zones)
        if not stacks:
            idle = 0.0
            start = None
            continue
        if start is None:
            start = sample.t
        idle += step
        if idle >= zones["remainder_sec"] and start is not None:
            return Hit(
                rule_id="fit_and_worn_mixed",
                t=start,
                observation="Счётная машина разделила пачку, а кассир соединила части, не проверив отложенную под ультрафиолетом.",
            )
    return None


def _count_without_client(samples: list[Sample], zones: dict) -> Hit | None:
    present = 0.0
    seen = False
    run = 0.0
    start: float | None = None
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        if _client_at_window(sample, zones):
            present += step
            if present >= zones["client_present_sec"]:
                seen = True
            run = 0.0
            start = None
            continue
        present = 0.0
        if not seen or not _wrist_in_counter(sample):
            run = 0.0
            start = None
            continue
        if start is None:
            start = sample.t
        run += step
        if run >= zones["count_wrist_sec"] and start is not None:
            return Hit(
                rule_id="recount_not_in_client_presence",
                t=start,
                observation="Кассир считает деньги на машинке, когда клиента у окна уже нет.",
            )
    return None


def _stranger_behind_door(sample: Sample, door: Det, zones: dict) -> bool:
    """Someone stands in the doorway, farther from the camera than the person at the desk."""
    people = _people(sample, zones, min_cx=0.0)
    if len(people) < 2:
        return False
    seated = max(people, key=lambda person: person.cy)
    for person in people:
        if person is seated:
            continue
        if seated.cy - person.cy < 0.08:
            continue
        if _dist(person.cx, person.cy, door.cx, door.cy) <= 0.22:
            return True
    return False


def _person_at_desk(sample: Sample, zones: dict) -> bool:
    """A person close to the camera is at the desk, not only stepping through the door."""
    return any(person.cy >= 0.50 for person in _people(sample, zones, min_cx=0.0))


def _spread_times(threshold: float, last: float, marks: list[float]) -> list[float]:
    """Threshold, a middle working moment, and a late moment while the door is still open."""
    late_marks = [t for t in marks if t >= threshold + 1.5]
    late = late_marks[-1] if late_marks else max(last, threshold)
    middle_marks = [t for t in marks if threshold + 0.8 <= t <= late - 0.8]
    if middle_marks:
        target = (threshold + late) / 2.0
        middle = min(middle_marks, key=lambda t: abs(t - target))
    else:
        middle = (threshold + late) / 2.0
    picked: list[float] = []
    for moment in (threshold, middle, late):
        if all(abs(moment - previous) >= 0.45 for previous in picked):
            picked.append(moment)
    return picked


def _persist_door(samples: list[Sample], zones: dict, conf: float) -> Hit | None:
    door_conf = zones["door_conf"]
    open_run = 0.0
    open_start: float | None = None
    max_area = 0.0
    open_hit: float | None = None
    open_outsider = False
    outsider_run = 0.0
    saw_outsider = False
    episode_closed = False
    threshold_t: float | None = None
    last_open: float | None = None
    work_times: list[float] = []
    busy_run = 0.0
    busy_start: float | None = None
    busy_hit: float | None = None
    busy_closed = False
    busy_threshold: float | None = None
    busy_last: float | None = None
    busy_work: list[float] = []
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        doors = _objs(sample, "door_open", door_conf)
        if doors:
            leaf = max(doors, key=lambda det: det.conf)
            if open_start is None:
                open_start = sample.t
                max_area = 0.0
                outsider_run = 0.0
                saw_outsider = False
            open_run += step
            max_area = max(max_area, leaf.area)
            if _stranger_behind_door(sample, leaf, zones):
                outsider_run += step
                if outsider_run >= 1.0:
                    saw_outsider = True
            else:
                outsider_run = 0.0
            need = zones["door_wide_sec"] if max_area >= zones["door_wide_area"] else zones["door_persist_sec"]
            if open_run >= need and open_start is not None:
                if open_hit is None:
                    open_hit = open_start
                    threshold_t = sample.t
                if saw_outsider:
                    open_outsider = True
            if threshold_t is not None and not episode_closed:
                last_open = sample.t
                if _person_at_desk(sample, zones):
                    work_times.append(sample.t)
        else:
            if threshold_t is not None:
                episode_closed = True
            open_run = 0.0
            open_start = None
            max_area = 0.0
            outsider_run = 0.0
            saw_outsider = False
        firm = _objs(sample, "door_open", conf)
        busy = bool(firm and (_objs(sample, "cash_stack", conf) or _objs(sample, "safe_open", conf)))
        if busy:
            if busy_start is None:
                busy_start = sample.t
            busy_run += step
            if not busy_closed:
                busy_last = sample.t
                if _person_at_desk(sample, zones):
                    busy_work.append(sample.t)
            if busy_run >= zones["door_with_cash_sec"] and busy_hit is None and busy_start is not None:
                busy_hit = busy_start
                busy_threshold = sample.t
        else:
            if busy_threshold is not None:
                busy_closed = True
            busy_run = 0.0
            busy_start = None
    if open_hit is not None and (busy_hit is None or open_hit <= busy_hit):
        if open_outsider:
            text = "Дверь кассы долго открыта. За ней виден человек, который в кассу не входил."
        else:
            text = "Дверь кассы остаётся открытой после прохода."
        proof = _spread_times(threshold_t or open_hit, last_open or open_hit, work_times)
        return Hit(
            rule_id="cash_desk_door_open",
            t=proof[-1],
            observation=text,
            evidence_at=proof,
        )
    if busy_hit is not None:
        proof = _spread_times(busy_threshold or busy_hit, busy_last or busy_hit, busy_work)
        return Hit(
            rule_id="cash_desk_door_open",
            t=proof[-1],
            observation="Дверь кассы открыта, пока на столе лежат деньги или открыт сейф.",
            evidence_at=proof,
        )
    return None


def _nearest_wrist(sample: Sample, cx: float, cy: float) -> float:
    if not sample.wrists:
        return 9.0
    return min(_dist(x, y, cx, cy) for x, y in sample.wrists)


def _persist_safe(samples: list[Sample], zones: dict, conf: float) -> Hit | None:
    """An open safe is allowed only while a hand is inside it."""
    run = 0.0
    start: float | None = None
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        safes = [
            det
            for det in _objs(sample, "safe_open", 0.6)
            if det.cy <= 0.42
        ]
        if not safes:
            run = 0.0
            start = None
            continue
        safe = max(safes, key=lambda det: det.conf)
        digging = any(safe.x1 <= x <= safe.x2 and safe.y1 <= y <= safe.y2 for x, y in sample.wrists)
        if digging:
            run = 0.0
            start = None
            continue
        if start is None:
            start = sample.t
        run += step
        if run >= zones["safe_hands_sec"] and start is not None:
            return Hit(
                rule_id="cash_left_unsecured",
                t=start,
                observation="Сейф открыт, а кассир занимается другим и не работает в нём.",
            )
    return None


def _persist_cash_idle(
    samples: list[Sample], zones: dict, anchor: tuple[float, float] | None, conf: float
) -> Hit | None:
    run = 0.0
    start: float | None = None
    prev: tuple[float, float] | None = None
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        stacks = [d for d in _desk_cash(sample, zones) if d.conf >= 0.4]
        if not stacks:
            run = 0.0
            start = None
            prev = None
            continue
        stack = max(stacks, key=lambda d: d.area)
        moved = prev is not None and _dist(stack.cx, stack.cy, prev[0], prev[1]) > zones["cash_idle_move"]
        prev = (stack.cx, stack.cy)
        working = (
            _hands_near(sample, stack, zones["cash_hands_dist"])
            or _wrist_in_counter(sample)
            or bool(_objs(sample, "safe_open", conf))
        )
        if moved or working:
            run = 0.0
            start = None
            continue
        if start is None:
            start = sample.t
        run += step
        if run >= zones["cash_hands_sec"] and start is not None:
            return Hit(
                rule_id="cash_left_unsecured",
                t=start,
                observation="Пачка лежит на столе без работы: кассир её не считает и не убирает.",
            )
    return None


def _labeled_phone(samples: list[Sample], zones: dict) -> Hit | None:
    run = 0.0
    start: float | None = None
    for index, sample in enumerate(samples):
        step = _dt(samples, index)
        if _objs(sample, "phone", zones["phone_conf"]):
            if start is None:
                start = sample.t
            run += step
            if run >= zones["phone_persist_sec"] and start is not None:
                return Hit(
                    rule_id="phone_with_camera_in_cash_desk",
                    t=start,
                    observation="В зоне кассы виден смартфон.",
                )
        else:
            run = 0.0
            start = None
    return None


def hits_to_violations(video_path: Path, hits: list[Hit], rules: dict) -> list[Violation]:
    violations: list[Violation] = []
    for hit in hits:
        rule = rules.get(hit.rule_id)
        if not rule:
            continue
        frames = evidence_frames(
            video_path,
            hit.t,
            n=settings.evidence_frame_count,
            span_sec=settings.evidence_span_sec,
            times=hit.evidence_at,
        )
        if not frames:
            continue
        legal = rule.get("legal_source") or {}
        basis = legal.get("basis") or ""
        description = hit.observation if not basis else f"{hit.observation} Основание: {basis}"
        violations.append(
            Violation(
                type=rule["title"],
                timecode_sec=round(float(hit.t), 2),
                description=description,
                evidence_frames=[
                    EvidenceFrame(timestamp_sec=frame.timestamp_sec, image_base64=frame.image_base64)
                    for frame in frames[:3]
                ],
                rule_document=legal.get("document") or None,
                rule_section=legal.get("section") or None,
            )
        )
    return violations


class RulesAnalyzer:
    """Detector plus one rule per catalog id that we can see. File name is ignored."""

    def __init__(self) -> None:
        self.model = load_detector()
        self.zones = load_zones()
        catalog_path = ROOT_DIR / "app" / "rules" / "catalog.json"
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.rules = {item["id"]: item for item in data.get("violation_types", [])}

    def analyze(self, video_path: Path, original_name: str | None = None) -> tuple[str, list[Violation]]:
        _ = original_name
        samples = collect_samples(video_path, fps=settings.rules_fps)
        hits = decide(samples, self.zones)
        if settings.vlm_enabled and hits:
            from app.services.vlm_verify import confirm_hits

            hits = confirm_hits(video_path, hits, self.rules)
        violations = hits_to_violations(video_path, hits, self.rules)
        if violations:
            return "violation", violations
        return "no_violation", []
