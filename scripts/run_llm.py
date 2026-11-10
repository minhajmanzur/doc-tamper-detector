"""Have Claude review the classically flagged regions on a benchmark subset.

    python scripts/run_llm.py --dry-run        # no API calls: save what the model would see
    python scripts/run_llm.py                  # test split: every forged image + as many authentic
    python scripts/run_llm.py --all            # whole test split (218 images)

Verdicts are cached per image in results/cache/llm-<provider>/, so reruns only pay for new images.
Needs GEMINI_API_KEY (free tier) or ANTHROPIC_API_KEY in .env; set TAMPER_PROVIDER to force one.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os

import cv2
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")          # before the provider modules read TAMPER_MODEL

from tamper import forensics as fx
from tamper import llm
from tamper.dataset import load_split


def provider() -> str:
    """TAMPER_PROVIDER wins; otherwise use whichever key is present (Claude first)."""
    p = os.environ.get("TAMPER_PROVIDER", "").lower()
    if p in ("claude", "gemini"):
        return p
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    sys.exit("No API key found. Put GEMINI_API_KEY (free tier) or ANTHROPIC_API_KEY in .env")


def pick(split: str, everything: bool, limit: int | None, seed: int = 0) -> list:
    samples = load_split(split)
    if not everything:
        forged = [s for s in samples if s.forged]
        authentic = [s for s in samples if not s.forged]
        random.Random(seed).shuffle(authentic)
        samples = forged + authentic[: len(forged)]
    samples.sort(key=lambda s: s.name)
    random.Random(seed).shuffle(samples)       # so --limit keeps the class mix
    return samples[:limit] if limit else samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--all", action="store_true", help="every image in the split, not the balanced subset")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, help="parallel requests (default 4 for Claude, 1 for Gemini's free tier)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    samples = pick(args.split, args.all, args.limit)

    if args.dry_run:
        out = ROOT / "results" / "debug" / "llm_input"
        out.mkdir(parents=True, exist_ok=True)
        for s in samples[:6]:
            ev = llm.Evidence(s.path, fx.analyze(s.path))
            cv2.imwrite(str(out / f"{s.name}_page.jpg"), ev.marked_page())
            for i, r in enumerate(ev.result.regions, 1):
                cv2.imwrite(str(out / f"{s.name}_R{i}.jpg"), ev.crop(*ev.region_box_on_page(r)))
            n_img = 1 + len(ev.result.regions)
            print(f"{s.name}: forged={s.forged} regions={len(ev.result.regions)} images_sent={n_img}")
        print(f"previews in {out}")
        return

    prov = provider()
    if prov == "gemini":
        from google import genai
        from google.genai import errors as gerrors
        from tamper import llm_gemini as backend
        client = genai.Client()
        api_errors: tuple = (gerrors.APIError,)
    else:
        import anthropic
        backend = llm
        client = anthropic.Anthropic(max_retries=4)
        api_errors = (anthropic.APIStatusError, anthropic.APIConnectionError)
    workers = args.workers or (1 if prov == "gemini" else 4)
    cache = ROOT / "results" / "cache" / f"llm-{prov}-{backend.MODEL}-{llm.PROMPT_VERSION}"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"provider: {prov}, model: {backend.MODEL}, prompt {llm.PROMPT_VERSION}, {len(samples)} receipts, {workers} worker(s)", flush=True)

    quota_gone = getattr(backend, "DailyQuotaExceeded", None) or ()
    stop = {"flag": False}

    def work(s) -> tuple[str, dict]:
        cached = cache / f"{s.name}.json"
        if cached.exists():
            return s.name, json.loads(cached.read_text())
        if stop["flag"]:
            return s.name, {"error": "skipped: daily quota reached"}
        try:
            verdict = backend.review(client, s.path, fx.analyze(s.path))
        except quota_gone:
            stop["flag"] = True
            return s.name, {"error": "skipped: daily quota reached"}
        except api_errors as e:
            code = getattr(e, "status_code", None) or getattr(e, "code", None)
            if code in (401, 403):
                raise                              # a bad key fails every item; stop now
            return s.name, {"error": f"api_{code}: {str(e)[:300]}"}
        if "error" not in verdict:                 # never cache failures
            cached.write_text(json.dumps(verdict, indent=1))
        return s.name, verdict

    results, tokens_in, tokens_out = {}, 0, 0
    with ThreadPoolExecutor(workers) as pool:
        for i, (name, v) in enumerate(pool.map(work, samples), 1):
            results[f"{args.split}/{name}"] = v
            u = v.get("usage", {})
            tokens_in += u.get("input_tokens", 0)
            tokens_out += u.get("output_tokens", 0)
            status = v.get("error") or f"{v['verdict']} p={v['fraud_probability']:.2f}"
            print(f"[{i}/{len(samples)}] {name}: {status}", flush=True)

    out = ROOT / "results" / ("llm.json" if llm.PROMPT_VERSION == "v2" else f"llm_{llm.PROMPT_VERSION}.json")
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged.update({k: v for k, v in results.items() if "error" not in v or k not in merged})
    out.write_text(json.dumps(merged, indent=1))
    errors = sum("error" in v for v in results.values())
    if stop["flag"]:
        print("Daily free-tier quota reached. Finished verdicts are cached; rerun the same command after the quota resets.")
    print(f"wrote {out}: {len(results) - errors} verdicts, {errors} errors, "
          f"{tokens_in:,} input / {tokens_out:,} output tokens (cached verdicts count their original usage)")


if __name__ == "__main__":
    main()
