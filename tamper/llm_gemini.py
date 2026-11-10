"""Gemini version of the region-review step (works on the free API tier).

Same evidence, same system prompt and same verdict schema as tamper/llm.py.
The free tier is capped per day per model (20 requests on the full Flash
models when this was written), so the default here is one request per receipt
on Flash-Lite: the model gets the page and every flagged crop up front and
answers in the verdict schema, with no zoom round-trips. Set TAMPER_ZOOM=1 to
use the multi-turn zoom agent instead on a key with more quota.
"""
from __future__ import annotations

import base64
import json
import os
import time

from google import genai
from google.genai import errors, types

from . import forensics as fx
from .llm import MAX_TURNS, PROMPT_VERSION, SYSTEM, VERDICT_SCHEMA, ZOOM_TOOL, Evidence, build_first_message

MODEL = os.environ.get("TAMPER_MODEL", "gemini-3.5-flash-lite")
ZOOM = os.environ.get("TAMPER_ZOOM", "0") == "1"


class DailyQuotaExceeded(RuntimeError):
    """The per-day free-tier cap for MODEL is used up; retrying today cannot help."""

INSTRUCTIONS = SYSTEM + ("\n\nYou act only through function calls. Call zoom to inspect an area; "
                         "when you have seen enough, call submit_verdict exactly once.")

TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(name="zoom", description=ZOOM_TOOL["description"], parameters_json_schema=ZOOM_TOOL["input_schema"]),
    types.FunctionDeclaration(name="submit_verdict", description="Deliver the final assessment of the document.",
                              parameters_json_schema=VERDICT_SCHEMA),
])]


def _to_parts(content: list[dict]) -> list[types.Part]:
    """Convert the Anthropic-style content blocks from build_first_message."""
    parts = []
    for block in content:
        if block["type"] == "text":
            parts.append(types.Part.from_text(text=block["text"]))
        else:
            parts.append(types.Part.from_bytes(data=base64.standard_b64decode(block["source"]["data"]), mime_type="image/jpeg"))
    return parts


def _jpeg_bytes(img) -> bytes:
    import cv2
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def _generate(client: genai.Client, contents: list, config: types.GenerateContentConfig):
    """generate_content with retries suited to the free tier.

    503 means the shared endpoint is overloaded and usually clears in seconds,
    so retry soon and often. 429 is a per-minute quota, so wait it out.
    """
    for attempt in range(12):
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except errors.APIError as e:
            if e.code == 429 and "PerDay" in str(getattr(e, "details", "")) + str(e):
                raise DailyQuotaExceeded(MODEL) from e
            if e.code not in (429, 500, 503) or attempt == 11:
                raise
            wait = 65 if e.code == 429 else min(4 + 3 * attempt, 20)
            print(f"  API {e.code}, retrying in {wait}s", flush=True)
            time.sleep(wait)


def _finish(verdict: dict, usage: dict, ev: Evidence) -> dict:
    missing = [k for k in VERDICT_SCHEMA["required"] if k not in verdict]
    if missing or verdict.get("verdict") not in ("forged", "authentic"):
        return {"error": f"bad_verdict: missing {missing}", "usage": usage, "model": MODEL}
    verdict["fraud_probability"] = float(min(1.0, max(0.0, verdict["fraud_probability"])))
    verdict.update(usage=usage, model=MODEL, requested_model=MODEL, page_scale=ev.page_scale, zoom_enabled=ZOOM, prompt_version=PROMPT_VERSION)
    return verdict


def review_single(client: genai.Client, ev: Evidence) -> dict:
    """One request: all evidence in, schema-constrained verdict out."""
    config = types.GenerateContentConfig(system_instruction=SYSTEM.replace(
        "Use the zoom tool when you need native resolution to judge glyph shapes or to read small text; the page "
        "image you start with is downscaled.", "The page image is downscaled; the region crops are native resolution."),
        response_mime_type="application/json", response_json_schema=VERDICT_SCHEMA, max_output_tokens=16000)
    response = _generate(client, [types.Content(role="user", parts=_to_parts(build_first_message(ev)))], config)
    um = response.usage_metadata
    usage = {"input_tokens": (um.prompt_token_count or 0) if um else 0,
             "output_tokens": ((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0,
             "cache_read_input_tokens": 0, "zooms": 0}
    if not response.text:
        reason = getattr(response.prompt_feedback, "block_reason", None) or (response.candidates[0].finish_reason if response.candidates else "empty")
        return {"error": f"blocked: {reason}", "usage": usage, "model": MODEL}
    try:
        return _finish(json.loads(response.text), usage, ev)
    except json.JSONDecodeError:
        return {"error": "invalid_json", "usage": usage, "model": MODEL}


def review(client: genai.Client, path: str, result: fx.ForensicResult) -> dict:
    ev = Evidence(path, result)
    if not ZOOM:
        return review_single(client, ev)
    contents: list = [types.Content(role="user", parts=_to_parts(build_first_message(ev)))]
    config = types.GenerateContentConfig(
        system_instruction=INSTRUCTIONS,
        tools=TOOLS,
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        max_output_tokens=16000,
    )
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "zooms": 0}
    for turn in range(MAX_TURNS):
        response = _generate(client, contents, config)
        um = response.usage_metadata
        if um:
            usage["input_tokens"] += um.prompt_token_count or 0
            usage["output_tokens"] += (um.candidates_token_count or 0) + (um.thoughts_token_count or 0)
        if not response.candidates or response.candidates[0].content is None:
            reason = getattr(response.prompt_feedback, "block_reason", None) or (response.candidates[0].finish_reason if response.candidates else "empty")
            return {"error": f"blocked: {reason}", "usage": usage, "model": MODEL}
        calls = response.function_calls or []
        if not calls:
            return {"error": f"no_function_call ({response.candidates[0].finish_reason})", "usage": usage, "model": MODEL}

        final = next((c for c in calls if c.name == "submit_verdict"), None)
        if final is not None:
            return _finish(dict(final.args), usage, ev)

        contents.append(response.candidates[0].content)      # keeps thought signatures intact
        parts = []
        for c in calls:
            usage["zooms"] += 1
            try:
                a = c.args
                img = ev.crop(int(a["x"]), int(a["y"]), int(a["width"]), int(a["height"]))
                parts.append(types.Part.from_function_response(
                    name=c.name, response={"result": "Crop attached: native resolution on the left, ELA heat map on the right."},
                    parts=[types.FunctionResponsePart.from_bytes(data=_jpeg_bytes(img), mime_type="image/jpeg")]))
            except (ValueError, KeyError, TypeError) as e:
                parts.append(types.Part.from_function_response(name=c.name, response={"error": str(e)}))
        if turn == MAX_TURNS - 2:
            parts.append(types.Part.from_text(text="That was the last zoom available. Call submit_verdict now."))
        contents.append(types.Content(role="user", parts=parts))
    return {"error": "turn_limit", "usage": usage, "model": MODEL}
