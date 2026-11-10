"""Run the OpenCV forensics over the whole dataset and cache features + regions.

    python scripts/run_classical.py            # all splits -> results/classical.json
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tamper import forensics as fx
from tamper.dataset import load_split

OUT = Path(__file__).resolve().parent.parent / "results" / "classical.json"


def work(path: str) -> dict:
    r = fx.analyze(path)
    return {"features": r.features, "regions": [g.to_dict() for g in r.regions],
            "copy_clusters": [list(c) for c in r.copy_pairs[:40]]}


def main() -> None:
    samples = [s for split in ("train", "val", "test") for s in load_split(split)]
    out = {}
    with ProcessPoolExecutor(max_workers=4) as pool:   # 35 MP scans need ~1.5 GB each
        for i, (s, res) in enumerate(zip(samples, pool.map(work, [s.path for s in samples], chunksize=8)), 1):
            out[f"{s.split}/{s.name}"] = res
            if i % 100 == 0:
                print(f"{i}/{len(samples)}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({len(out)} images)")


if __name__ == "__main__":
    main()
