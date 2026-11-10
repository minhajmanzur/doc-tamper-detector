"""LLM review of classically flagged regions.

The classical stage answers "where do the pixels look odd?". This stage answers
"is that oddness an edit that changes what the document says, or something
harmless?" Claude gets the page, the flagged regions with their forensic
signals, and a zoom tool to inspect anything else at native resolution, then
returns a structured verdict with a per-region explanation.
"""
from __future__ import annotations

import base64
import json
import os

import anthropic
import cv2
import numpy as np

from . import forensics as fx

MODEL = os.environ.get("TAMPER_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("TAMPER_EFFORT", "medium")
PAGE_SIDE = 1400          # longest side of the page image sent to the model
MAX_TURNS = 6

SYSTEM_V1 = """You review scanned documents (receipts, invoices, IDs) for a fraud team at an insurer. \
A classical image-forensics pipeline has already run on the page and flagged regions whose pixels look \
statistically unusual. Your job is the part the pipeline cannot do: decide what each flagged region means.

What counts as a forgery here: an edit that changes what the document claims. Typical cases are an altered \
total or line price, a changed date, a swapped company name or address, text copied from elsewhere on the \
page, or text retyped in a font that almost matches. These are made with image editors, so look for glyphs \
that are sharper, cleaner, differently weighted, misaligned with the baseline, or spaced differently from \
their neighbours, for background patches that are flatter or a different shade than the paper around them, \
and for content that does not add up (line items versus total, tax arithmetic, implausible dates).

What does not count: these documents were handled by real people before scanning. Handwritten notes, \
circled totals, highlighter, stamps, staple marks, folds, stains, thermal-print fading and scanner noise are \
normal and leave the document authentic even though they often trip the forensic signals. Bold headlines, \
logos and barcodes also trip them. A flagged region is a lead, not a finding, and the pipeline misses edits \
too, so check the fields a fraudster would target even when nothing there is flagged.

The forensic signals:
- ela: error level analysis. The region responds differently to JPEG recompression than the rest of the \
inked page, which suggests a different compression history.
- noise: the scan noise level in the region differs from the page.
- copymove: nearly pixel-identical patches exist at two locations, which printing and scanning does not \
produce by itself.

Use the zoom tool when you need native resolution to judge glyph shapes or to read small text; the page \
image you start with is downscaled. Base the verdict on what you can see. When the evidence is thin, say so \
and let fraud_probability reflect it rather than guessing confidently."""

_COPYMOVE_V1 = """- copymove: nearly pixel-identical patches exist at two locations, which printing and scanning does not \
produce by itself.
"""

# v2: under v1 the model overruled copy-move evidence because a pasted glyph looks normal.
# This version says what the signal physically means and when it may be set aside.
_COPYMOVE_V2 = """- copymove: a patch of inked pixels in this region is identical, pixel for pixel, to a patch somewhere \
else on the page, and several neighbouring blocks share the same offset. This is the strongest signal of the \
three and it is not a judgement about appearance. A printer and a scanner never produce two identical copies \
of a glyph: toner spread, paper texture and sensor noise make every printed character differ slightly from \
every other, even the same digit in the same font. Exact duplicates mean the pixels were copied in an image \
editor. A pasted digit therefore looks completely normal, with the right font, weight and alignment, because \
it is a real digit from the same receipt. Looking normal is what a copy-paste forgery looks like, so it is not \
a reason to dismiss this signal. Repeated characters, fonts or receipt structure do not explain it either; \
those were already ruled out by requiring exact identity.
  When copymove fired on a region that contains meaningful content (a price, quantity, total, date, time, \
number or name), assess it as content_edit and give the document a high fraud_probability, even if you cannot \
see anything wrong. Set it aside only when the duplicated pixels carry no meaning, such as a solid black \
redaction bar, a blank or saturated area, a decorative border or a repeated logo element.
- ela and noise are weak on this kind of scan and fire often on bold print, stamps and handwriting. On their \
own they justify a closer look, not a verdict.
"""

assert _COPYMOVE_V1 in SYSTEM_V1
SYSTEM_V2 = SYSTEM_V1.replace(_COPYMOVE_V1, _COPYMOVE_V2)

PROMPT_VERSION = os.environ.get("TAMPER_PROMPT", "v2")
SYSTEM = {"v1": SYSTEM_V1, "v2": SYSTEM_V2}[PROMPT_VERSION]

ZOOM_TOOL = {
    "name": "zoom",
    "description": (
        "Return a native-resolution crop of the document next to its error-level heat map. "
        "Coordinates are pixels in the page image you were shown. Use it to inspect glyph shapes, "
        "alignment and background texture, or to read small text such as line items and totals."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Left edge"},
            "y": {"type": "integer", "description": "Top edge"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
        },
        "required": ["x", "y", "width", "height"],
        "additionalProperties": False,
    },
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "regions": {
            "type": "array",
            "description": "One entry per flagged region, in the order given.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Region id, e.g. R1"},
                    "assessment": {
                        "type": "string",
                        "enum": ["content_edit", "benign_annotation", "print_or_scan_artifact", "nothing_unusual"],
                    },
                    "explanation": {"type": "string", "description": "What is in the region and why it got this assessment."},
                },
                "required": ["id", "assessment", "explanation"],
                "additionalProperties": False,
            },
        },
        "unflagged_findings": {
            "type": "array",
            "description": "Suspected edits the pipeline did not flag. Empty when there are none.",
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "explanation": {"type": "string"},
                },
                "required": ["x", "y", "width", "height", "explanation"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string", "description": "Two or three sentences a claims handler can act on."},
        "fraud_probability": {"type": "number", "description": "0 to 1: probability the document content was altered."},
        "verdict": {"type": "string", "enum": ["forged", "authentic"]},
    },
    "required": ["regions", "unflagged_findings", "summary", "fraud_probability", "verdict"],
    "additionalProperties": False,
}


def _b64_jpeg(img: np.ndarray, quality: int = 88) -> dict:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    data = base64.standard_b64encode(enc.tobytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


class Evidence:
    """Everything the model may look at for one document."""

    def __init__(self, path: str, result: fx.ForensicResult):
        self.full = cv2.imread(path, cv2.IMREAD_COLOR)                    # original resolution
        self.analysis, _ = fx.load_for_analysis(path)
        self.result = result
        h, w = self.full.shape[:2]
        self.page_scale = min(1.0, PAGE_SIDE / max(h, w))                 # page px / original px
        self.page = cv2.resize(self.full, (round(w * self.page_scale), round(h * self.page_scale)), interpolation=cv2.INTER_AREA)
        self.ela_heat = fx.heat_overlay(self.analysis, result.maps["ela"], vmax=2 * fx.THRESH["ela"])

    def marked_page(self) -> np.ndarray:
        return fx.draw_regions(self.page, self.result.regions, self.page_scale)

    def crop(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Crop given in page-image pixels -> native crop beside its ELA heat map."""
        s = 1.0 / self.page_scale
        H, W = self.full.shape[:2]
        x0, y0 = max(0, int(x * s)), max(0, int(y * s))
        x1, y1 = min(W, int((x + w) * s)), min(H, int((y + h) * s))
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError("crop is empty or outside the page")
        native = self.full[y0:y1, x0:x1]
        a = self.result.scale
        heat = self.ela_heat[int(y0 * a):int(y1 * a), int(x0 * a):int(x1 * a)]
        heat = cv2.resize(heat, (native.shape[1], native.shape[0]), interpolation=cv2.INTER_NEAREST)
        both = np.hstack([native, np.full((native.shape[0], 8, 3), 255, np.uint8), heat])
        m = max(both.shape[:2])
        if m < 700:                                # single-glyph crops are unreadable at 1:1
            both = cv2.resize(both, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        elif m > 1500:
            both = cv2.resize(both, None, fx=1500 / m, fy=1500 / m, interpolation=cv2.INTER_AREA)
        return both

    def region_box_on_page(self, r: fx.Region, pad: int = 60) -> tuple[int, int, int, int]:
        s = self.page_scale
        return (int(r.x * s) - pad, int(r.y * s) - pad, int(r.w * s) + 2 * pad, int(r.h * s) + 2 * pad)


def build_first_message(ev: Evidence) -> list[dict]:
    ph, pw = ev.page.shape[:2]
    content: list[dict] = [
        {"type": "text", "text": f"Document page, {pw}x{ph} px. Flagged regions are outlined in red and labelled."},
        _b64_jpeg(ev.marked_page()),
    ]
    if not ev.result.regions:
        content.append({"type": "text", "text": "The forensic pipeline flagged no regions on this page."})
    for i, r in enumerate(ev.result.regions, 1):
        x, y, w, h = ev.region_box_on_page(r)
        content.append({
            "type": "text",
            "text": (f"Region R{i}: page box x={x} y={y} w={w} h={h}; signals fired: {', '.join(r.signals) or 'none'}; "
                     f"evidence score {r.score:.2f} (1.0 = flagging threshold). Native crop on the left, ELA heat map on the right."),
        })
        try:
            content.append(_b64_jpeg(ev.crop(x, y, w, h)))
        except ValueError:
            content.append({"type": "text", "text": "(crop unavailable)"})
        if PROMPT_VERSION != "v1" and "copymove" in r.signals:
            # v2: restate the physical meaning next to the crop, where the model is deciding
            content.append({"type": "text", "text": (
                f"R{i} contains pixels that are an exact copy of pixels elsewhere on this page. That does not happen "
                "in printing or scanning, so something here was pasted, however normal it looks. Decide what the "
                "pasted content is: if it is part of a price, total, quantity, date, number or name, this is a "
                "content_edit.")})
    content.append({"type": "text", "text": "Assess each flagged region, check the fields worth forging, and give your verdict."})
    return content


def review(client: anthropic.Anthropic, path: str, result: fx.ForensicResult) -> dict:
    """Run the agent loop for one document and return the parsed verdict plus usage."""
    ev = Evidence(path, result)
    messages: list[dict] = [{"role": "user", "content": build_first_message(ev)}]
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "zooms": 0}
    for turn in range(MAX_TURNS):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            # A page about forged documents can trip a safety classifier; let the API
            # reroute a declined request to another model instead of failing the item.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=[ZOOM_TOOL],
            messages=messages,
        )
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
        usage["cache_read_input_tokens"] += response.usage.cache_read_input_tokens or 0

        if response.stop_reason == "refusal":
            return {"error": "refusal", "usage": usage, "model": response.model}
        if response.stop_reason == "max_tokens":
            return {"error": "max_tokens", "usage": usage, "model": response.model}
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                usage["zooms"] += 1
                try:
                    a = block.input
                    img = ev.crop(int(a["x"]), int(a["y"]), int(a["width"]), int(a["height"]))
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": [_b64_jpeg(img)]})
                except (ValueError, KeyError, TypeError) as e:
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": f"Error: {e}", "is_error": True})
            if turn == MAX_TURNS - 2:
                results.append({"type": "text", "text": "That was the last zoom available. Give your verdict now."})
            messages.append({"role": "user", "content": results})
            continue

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            return {"error": "no_text", "usage": usage, "model": response.model}
        verdict = json.loads(text)
        verdict["fraud_probability"] = float(min(1.0, max(0.0, verdict["fraud_probability"])))
        verdict["usage"] = usage
        verdict["model"] = response.model       # differs from MODEL when a fallback served the request
        verdict["requested_model"] = MODEL
        verdict["prompt_version"] = PROMPT_VERSION
        verdict["page_scale"] = ev.page_scale
        return verdict
    return {"error": "turn_limit", "usage": usage}
