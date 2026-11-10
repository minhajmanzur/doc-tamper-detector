"""Benchmark the classical signals, and compare them with the LLM when verdicts exist.

    python scripts/evaluate.py     # reads results/classical.json (+ results/llm.json), writes results/report.md

Protocol: the fusion model is fitted on train, its decision threshold is picked
on val, and every number in the report is from the held-out test split.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tamper import forensics as fx
from tamper.dataset import TECHNIQUES, load_split, overlaps

RES = ROOT / "results"
FIG = RES / "figures"
FEATS = ["ela_max", "ela_p99", "noise_max", "noise_p99", "copymove_blocks", "copymove_clusters", "n_regions"]
SINGLE = {"ELA": "ela_max", "Noise inconsistency": "noise_max", "Copy-move": "copymove_blocks"}
INK, MUTED, C1, C2, C3 = "#1f2933", "#7b8794", "#2a78d6", "#d9822b", "#2e9e6b"


def X_of(samples, classical):
    rows = []
    for s in samples:
        f = classical[f"{s.split}/{s.name}"]["features"]
        rows.append([np.log1p(f[k]) if k.startswith("copymove") else f[k] for k in FEATS])
    return np.array(rows)


def prf(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return p, r, f


def boot_auc(y, score, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], score[i]))
    return np.percentile(vals, [2.5, 97.5])


def region_hits(sample, regions, pad=8):
    """Indices of proposed regions that touch an annotated altered area (or a copy source)."""
    return [i for i, r in enumerate(regions)
            if any(overlaps((r["x"], r["y"], r["w"], r["h"]), b, pad) for b in sample.boxes + sample.sources)]


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def save_example(sample, classical, path, title):
    img, scale = fx.load_for_analysis(sample.path)
    regs = [fx.Region(**r) for r in classical[f"{sample.split}/{sample.name}"]["regions"]]
    out = fx.draw_regions(img, regs, scale)
    for x, y, w, h in sample.boxes:                       # ground truth in green
        cv2.rectangle(out, (int(x * scale) - 3, int(y * scale) - 3), (int((x + w) * scale) + 3, int((y + h) * scale) + 3), (60, 170, 40), 2)
    cv2.imwrite(str(path), out)


def main() -> None:
    classical = json.loads((RES / "classical.json").read_text())
    train, val, test = (load_split(s) for s in ("train", "val", "test"))
    y_tr, y_va, y_te = (np.array([s.forged for s in part], int) for part in (train, val, test))
    FIG.mkdir(parents=True, exist_ok=True)
    L = ["# Benchmark report", "",
         f"Dataset: Find it again! (L3i, ICDAR 2023). Test split: {len(test)} receipts, {int(y_te.sum())} forged. "
         f"Fusion fitted on train ({len(train)}), threshold chosen on val ({len(val)}).", ""]

    # ---------------------------------------------------------------- classical
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    model.fit(X_of(train, classical), y_tr)
    p_va, p_te = model.predict_proba(X_of(val, classical))[:, 1], model.predict_proba(X_of(test, classical))[:, 1]
    cands = np.unique(p_va)
    thr = float(max(cands, key=lambda t: prf(y_va, p_va >= t)[2]))
    pred_te = (p_te >= thr).astype(int)

    rows, curves = [], {}
    for label, key in SINGLE.items():
        sc = np.array([classical[f"{s.split}/{s.name}"]["features"][key] for s in test])
        lo, hi = boot_auc(y_te, sc)
        rows.append([label, f"{roc_auc_score(y_te, sc):.3f}", f"{lo:.2f} to {hi:.2f}", "", "", ""])
        curves[label] = roc_curve(y_te, sc)
    lo, hi = boot_auc(y_te, p_te)
    p, r, f = prf(y_te, pred_te)
    rows.append(["**Fused (logistic)**", f"{roc_auc_score(y_te, p_te):.3f}", f"{lo:.2f} to {hi:.2f}", f"{p:.2f}", f"{r:.2f}", f"{f:.2f}"])
    curves["Fused"] = roc_curve(y_te, p_te)
    L += ["## Classical signals, image level", "", md_table(["Signal", "ROC AUC", "95% CI", "Precision", "Recall", "F1"], rows), "",
          "AUC intervals are bootstrap percentiles over the test images. Precision, recall and F1 use the validation-chosen threshold.", ""]

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for (label, (fpr, tpr, _)), c in zip(curves.items(), (C2, MUTED, C3, C1)):
        ax.plot(fpr, tpr, color=c, lw=2.4 if label == "Fused" else 1.6, label=label)
    ax.plot([0, 1], [0, 1], color="#cbd2d9", lw=1, ls="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Classical signals on the test split")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "roc_classical.png", dpi=160); plt.close(fig)
    L += ["![ROC curves](figures/roc_classical.png)", ""]

    # localisation and per-technique recall
    forged_te = [s for s in test if s.forged]
    loc = sum(bool(region_hits(s, classical[f"test/{s.name}"]["regions"])) for s in forged_te)
    tech = defaultdict(lambda: [0, 0, 0])
    for s, pr in zip(test, pred_te):
        if not s.forged:
            continue
        hit = bool(region_hits(s, classical[f"test/{s.name}"]["regions"]))
        for t in set(s.techniques) or {"unlabelled"}:
            tech[t][0] += 1; tech[t][1] += int(pr); tech[t][2] += int(hit)
    L += ["## Where the classical stage works and where it does not", "",
          f"Localisation: on {loc} of {len(forged_te)} forged test receipts, at least one of the top-5 proposed regions touches an annotated altered area or its copy source.", "",
          md_table(["Forgery technique", "Forged receipts", "Detected", "Localised"],
                   [[f"{t} ({TECHNIQUES.get(t, t)})", n, f"{d} ({d / n:.0%})", f"{h} ({h / n:.0%})"] for t, (n, d, h) in sorted(tech.items(), key=lambda kv: -kv[1][0])]), "",
          "A receipt with several techniques is counted under each.", ""]

    auth = [(s, pr) for s, pr in zip(test, pred_te) if not s.forged]
    groups = {"no annotation": lambda s: not (s.digital_annotation or s.handwritten_annotation),
              "handwritten annotation": lambda s: s.handwritten_annotation,
              "digital annotation": lambda s: s.digital_annotation}
    rows = []
    for g, fn in groups.items():
        sel = [pr for s, pr in auth if fn(s)]
        rows.append([g, len(sel), f"{sum(sel)} ({(sum(sel) / len(sel) if sel else 0):.0%})"])
    L += ["False alarms on authentic receipts, by the benign marks they carry:", "", md_table(["Authentic receipts with", "Count", "Flagged as forged"], rows), ""]

    # ---------------------------------------------------------------------- LLM
    llm_path = RES / "llm.json"
    verdicts = {k: v for k, v in (json.loads(llm_path.read_text()) if llm_path.exists() else {}).items()
                if k.startswith("test/") and "error" not in v}
    if not verdicts:
        L += ["## LLM stage", "", "Not run yet: `results/llm.json` has no verdicts. Add `ANTHROPIC_API_KEY` to `.env`, "
              "run `python scripts/run_llm.py`, then rerun this script to fill in the comparison and disagreement sections.", ""]
    else:
        idx = {s.name: i for i, s in enumerate(test)}
        sub = [s for s in test if f"test/{s.name}" in verdicts]
        y = np.array([s.forged for s in sub], int)
        c_prob = np.array([p_te[idx[s.name]] for s in sub]); c_pred = np.array([pred_te[idx[s.name]] for s in sub])
        l_prob = np.array([verdicts[f"test/{s.name}"]["fraud_probability"] for s in sub])
        l_pred = np.array([verdicts[f"test/{s.name}"]["verdict"] == "forged" for s in sub], int)
        rows = []
        methods = [("Classical fused", c_prob, c_pred), ("LLM review (prompt v2)", l_prob, l_pred)]
        v1_path = RES / "llm_v1.json"
        v1 = json.loads(v1_path.read_text()) if v1_path.exists() else {}
        if all(f"test/{s.name}" in v1 and "error" not in v1[f"test/{s.name}"] for s in sub):
            methods.append(("LLM review (prompt v1)", np.array([v1[f"test/{s.name}"]["fraud_probability"] for s in sub]),
                            np.array([v1[f"test/{s.name}"]["verdict"] == "forged" for s in sub], int)))
        for label, prob, pred in methods:
            p, r, f = prf(y, pred)
            lo, hi = boot_auc(y, prob)
            rows.append([label, f"{roc_auc_score(y, prob):.3f}", f"{lo:.2f} to {hi:.2f}", f"{p:.2f}", f"{r:.2f}", f"{f:.2f}", f"{(pred == y).mean():.2f}"])
        fb = sum(1 for s in sub if verdicts[f"test/{s.name}"].get("model") != verdicts[f"test/{s.name}"].get("requested_model"))
        models = ", ".join(sorted({str(verdicts[f"test/{s.name}"].get("requested_model")) for s in sub}))
        L += ["## Classical versus LLM", "",
              f"Subset reviewed by the LLM ({models}): {len(sub)} test receipts, {int(y.sum())} forged. Class balance here differs from the full test split, "
              "so precision is not comparable with the table above; both rows below are on this same subset.", "",
              md_table(["Method", "ROC AUC", "95% CI", "Precision", "Recall", "F1", "Accuracy"], rows), ""]
        if fb:
            L += [f"{fb} verdicts were served by a fallback model after the primary declined the request.", ""]

        cells = Counter()
        cases = defaultdict(list)
        for s, cp, lp, yy in zip(sub, c_pred, l_pred, y):
            key = ("forged" if yy else "authentic", "both right" if cp == yy and lp == yy else "only classical right" if cp == yy
                   else "only LLM right" if lp == yy else "both wrong")
            cells[key] += 1
            cases[key].append(s)
        cols = ["both right", "only classical right", "only LLM right", "both wrong"]
        L += ["### Agreement", "", md_table(["Ground truth"] + cols, [[g] + [cells[(g, c)] for c in cols] for g in ("forged", "authentic")]), ""]

        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        for yy, c, lab in ((0, C1, "authentic"), (1, C2, "forged")):
            ax.scatter(c_prob[y == yy], l_prob[y == yy], s=34, color=c, alpha=0.75, label=lab, edgecolor="white", linewidth=0.6)
        ax.axvline(thr, color="#cbd2d9", lw=1, ls="--"); ax.axhline(0.5, color="#cbd2d9", lw=1, ls="--")
        ax.set(xlabel="Classical fused score", ylabel="LLM fraud probability", title="Where the two stages disagree", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
        ax.legend(frameon=False, loc="center right"); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(FIG / "disagreement.png", dpi=160); plt.close(fig)
        L += ["![Disagreement scatter](figures/disagreement.png)", "",
              "Top-left and bottom-right quadrants are the disagreements; colour is the ground truth.", ""]

        # region level: what did the LLM make of each flagged region?
        reg = Counter()
        for s in sub:
            regions = classical[f"test/{s.name}"]["regions"]
            hits = set(region_hits(s, regions)) if s.forged else set()
            for i, a in enumerate(verdicts[f"test/{s.name}"].get("regions", [])[: len(regions)]):
                reg[("on an altered area" if i in hits else "not on an altered area", a["assessment"])] += 1
        kinds = ["content_edit", "benign_annotation", "print_or_scan_artifact", "nothing_unusual"]
        L += ["### Region level: how the LLM read the classical flags", "",
              md_table(["Flagged region is"] + kinds, [[g] + [reg[(g, k)] for k in kinds] for g in ("on an altered area", "not on an altered area")]), "",
              "First row, `content_edit` is the LLM confirming a true flag; the other columns are true flags it explained away. "
              "Second row, `content_edit` is the LLM endorsing a false flag; the other columns are false flags it correctly dismissed.", ""]

        L += ["### Disagreement cases", ""]
        titles = {("forged", "only classical right"): "Forged, caught by pixels, cleared by the LLM",
                  ("forged", "only LLM right"): "Forged, missed by pixels, caught by the LLM",
                  ("authentic", "only classical right"): "Authentic, passed by pixels, accused by the LLM",
                  ("authentic", "only LLM right"): "Authentic, flagged by pixels, cleared by the LLM",
                  ("forged", "both wrong"): "Forged, missed by both"}
        for key, title in titles.items():
            if not cases[key]:
                continue
            L += [f"**{title}** ({len(cases[key])})", ""]
            for s in cases[key][:4]:
                v = verdicts[f"test/{s.name}"]
                img = FIG / f"case_{s.name}"
                save_example(s, classical, img, title)
                tech_s = ", ".join(sorted(set(s.techniques))) or "none"
                L += [f"- `{s.name}` (techniques: {tech_s}; classical {p_te[idx[s.name]]:.2f}, LLM {v['fraud_probability']:.2f}) "
                      f"[overlay](figures/{img.name}). LLM: {v['summary']}"]
            L += [""]
        L += ["Overlays: red boxes are classical proposals, green boxes are ground-truth altered areas.", ""]

    # a few classical examples regardless of LLM
    L += ["## Example overlays", ""]
    order = np.argsort(-p_te)
    picks = [("Highest-scoring forged", next(i for i in order if y_te[i])), ("Highest-scoring authentic (false alarm)", next(i for i in order if not y_te[i])),
             ("Lowest-scoring forged (miss)", next(i for i in order[::-1] if y_te[i]))]
    for title, i in picks:
        s = test[i]
        save_example(s, classical, FIG / f"example_{s.name}", title)
        L += [f"- {title}: `{s.name}`, score {p_te[i]:.2f}, techniques {', '.join(sorted(set(s.techniques))) or 'none'}. [overlay](figures/example_{s.name})"]
    L += ["", f"Fusion threshold (from val): {thr:.3f}. Fusion coefficients (standardised): "
          + ", ".join(f"{k} {c:+.2f}" for k, c in zip(FEATS, model[-1].coef_[0])) + "."]

    (RES / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RES / "classical_scores.json").write_text(json.dumps(
        {"threshold": thr, "test": {s.name: {"score": float(p), "pred": int(d), "forged": bool(s.forged)} for s, p, d in zip(test, p_te, pred_te)}}, indent=1))
    print("\n".join(L[:22]))
    print(f"\nwrote {RES / 'report.md'}")


if __name__ == "__main__":
    main()
