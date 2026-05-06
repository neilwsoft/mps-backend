"""Gemini client for generating one-sentence error explanations and for
extracting handwritten math (line-by-line LaTeX) from an uploaded image.

If ``GEMINI_API_KEY`` is missing, callers fall back to deterministic
behavior (no explanation, or an empty extraction list).
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = """You are a math tutor for K-12 students.
A student is solving a problem step-by-step. Below is the problem, the
expected next step from the model solution, and what the student wrote.

Problem (LaTeX): {problem}
Expected step (LaTeX): {expected}
Student wrote (LaTeX): {submitted}

Explain in ONE short sentence (under 25 words, no LaTeX, no markdown)
what the student likely did wrong. If the student's line is correct in
value but doesn't match the expected step, say so plainly. Output only
the sentence."""


_EXTRACT_PROMPT = """You are an OCR system specialised in handwritten K-12 algebra.

The student is solving this problem:
  Problem (LaTeX): {problem}

The student's full model solution has these steps (for reference only —
the student's handwriting may not match exactly, that is fine):
{solution_hint}

Look at the attached image of the student's handwritten work. Each
visually distinct horizontal line in the image is ONE step of the
solution. Treat continued lines (line break inside what is clearly the
same expression) as a single line.

Convert each line into a LaTeX math expression:
  - Strip any leading "=" sign — return only the right-hand side of each line.
    The "=" can appear on the same line as an expression OR on its own
    little line/column to the left of an expression; in both cases ignore
    it and return only the expression.
  - If the FIRST handwritten line is just the student copying the
    original problem verbatim (no simplification yet), OMIT it — your
    output should start with the first actual solution step.
    A line counts as a "copy of the problem" if, ignoring whitespace and
    a leading "=", it is the same expression (or a trivially equivalent
    rearrangement of brackets) as the Problem above.
  - Use ^ for exponents, e.g. x^2.
  - Use {{ }} for grouping when a sub-expression spans more than one
    character, e.g. x^{{12}}.
  - Use \\frac{{a}}{{b}} for fractions.
  - Preserve curly braces {{ }} and brackets [ ] that the student wrote.
  - Do NOT add steps the student didn't write.
  - Do NOT solve the problem yourself.

Return STRICT JSON with this shape (no markdown, no commentary):
  {{ "lines": ["<latex line 1>", "<latex line 2>", ...] }}

If the image is unreadable or contains no math, return {{ "lines": [] }}."""


def explain_mistake(problem: str, expected: str, submitted: str) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai not installed; skipping Gemini explanation")
        return None

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=_PROMPT_TEMPLATE.format(
                problem=problem, expected=expected, submitted=submitted
            ),
        )
        text = (response.text or "").strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 — never crash the verify endpoint
        logger.warning("Gemini call failed: %s", exc)
        return None


def _strip_code_fence(text: str) -> str:
    """Gemini sometimes wraps JSON in ```json ... ``` despite our instructions."""
    fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    return text


def extract_handwritten_math(
    image_bytes: bytes,
    mime_type: str,
    problem_latex: str,
    solution_latex: list[str],
) -> list[str]:
    """Run Gemini Vision over a handwritten image and return one LaTeX line per
    visual line. Returns an empty list if Gemini is unavailable or the response
    can't be parsed — the caller decides how to surface that."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; cannot extract handwritten math")
        return []

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("google-genai not installed; cannot extract handwritten math")
        return []

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    solution_hint = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(solution_latex))
    prompt = _EXTRACT_PROMPT.format(
        problem=problem_latex,
        solution_hint=solution_hint or "  (no model solution available)",
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini vision call failed: %s", exc)
        return []

    raw = (response.text or "").strip()
    if not raw:
        return []

    cleaned = _strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Gemini vision response was not valid JSON: %r", raw[:200])
        return []

    lines = data.get("lines") if isinstance(data, dict) else None
    if not isinstance(lines, list):
        return []
    return [str(x).strip() for x in lines if str(x).strip()]
