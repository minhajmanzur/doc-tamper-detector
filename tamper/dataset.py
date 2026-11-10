"""Loader for the 'Find it again!' receipt forgery dataset (L3i, ICDAR 2023).

Each split has a CSV-like .txt: image, digital annotation, handwritten
annotation, forged, forgery annotations (a VIA-style dict with rectangles).
Rectangles marked 'Original area: yes' are the *source* of a copy, not an
altered area, so they are kept separately.
"""
from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "findit2"

# Technique codes used by the dataset authors
TECHNIQUES = {"CPI": "copy-paste inside", "CPO": "copy-paste outside", "IMI": "imitation (retyped)",
              "CUT": "cut / erase", "PIX": "pixel edit", "Other": "other"}


@dataclass
class Sample:
    split: str
    name: str
    path: str
    forged: bool
    digital_annotation: bool      # benign: typed/drawn notes added digitally
    handwritten_annotation: bool  # benign: pen marks made before scanning
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)   # altered areas (x, y, w, h)
    sources: list[tuple[int, int, int, int]] = field(default_factory=list)  # copy sources
    techniques: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


def _rect(shape: dict) -> tuple[int, int, int, int] | None:
    if shape.get("name") != "rect":
        return None
    return int(shape["x"]), int(shape["y"]), int(shape["width"]), int(shape["height"])


def load_split(split: str, root: Path = ROOT) -> list[Sample]:
    samples = []
    with open(root / f"{split}.txt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = Sample(split, row["image"], str(root / split / row["image"]), row["forged"].strip() == "1",
                       row["digital annotation"].strip() == "1", row["handwritten annotation"].strip() == "1")
            if not Path(s.path).exists():      # one val image is listed but absent from the archive
                continue
            raw = (row.get("forgery annotations") or "").strip()
            if s.forged and raw and raw != "0":
                try:
                    ann = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    ann = {}
                for reg in ann.get("regions", []):
                    box = _rect(reg.get("shape_attributes", {}))
                    attrs = reg.get("region_attributes", {})
                    if box is None:
                        continue
                    if str(attrs.get("Original area", "no")).lower() == "yes":
                        s.sources.append(box)
                        continue
                    s.boxes.append(box)
                    mod = attrs.get("Modified area", {})
                    s.techniques += [k for k, v in mod.items() if v and k != "None"] if isinstance(mod, dict) else []
                    if attrs.get("Entity type"):
                        s.entities.append(str(attrs["Entity type"]))
            samples.append(s)
    return samples


def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax - pad < bx + bw and bx - pad < ax + aw and ay - pad < by + bh and by - pad < ay + ah
