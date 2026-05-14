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


Locale = str  # expected values: "en" | "ko"


def _language_directive(locale: Locale) -> str:
    """A single line appended to every prompt to force the response language.
    Math notation (LaTeX) stays the same regardless of locale."""
    if locale == "ko":
        return (
            "RESPOND IN KOREAN (한국어). Use polite, natural classroom Korean "
            "appropriate for a K-12 student (e.g. -습니다/-세요 forms). Do not "
            "translate LaTeX math; keep formulas exactly as they are."
        )
    return "RESPOND IN ENGLISH. Use clear, kind K-12 classroom English."


_PROMPT_TEMPLATE = """You are a math tutor for K-12 students.
A student is solving a problem step-by-step. Below is the problem, the
expected next step from the model solution, and what the student wrote.

Problem (LaTeX): {problem}
Expected step (LaTeX): {expected}
Student wrote (LaTeX): {submitted}

Explain in ONE short sentence (under 25 words, no LaTeX, no markdown)
what the student likely did wrong. If the student's line is correct in
value but doesn't match the expected step, say so plainly. Output only
the sentence.

{lang}"""


_WHY_WRONG_PROMPT = """You are a kind, encouraging math tutor for K-12 students.
A student is solving a problem step-by-step.

Problem (LaTeX): {problem}
Expected next step (LaTeX): {expected}
Student wrote (LaTeX): {submitted}

In 2 to 3 short sentences (no LaTeX, no markdown, no emoji), tell the student:
  1. What operation they were supposed to do here.
  2. What likely went wrong (e.g. "you distributed the minus sign over only
     one term", "you forgot to combine the like terms", "the sign on x flipped").
  3. A nudge for what to try next — but DO NOT reveal the expected step itself.

Be specific and kind. If the student's line is value-equivalent to the expected
step but in a different form, say so plainly and ask them to write the form the
worksheet expects. Output only the sentences, nothing else.

{lang}"""


_EXTRACT_PROMPT = """You are a fast OCR system for handwritten K-12 algebra. Be terse and accurate.

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

For each line, also estimate your OCR confidence as a number between 0.0
and 1.0:
  - 0.9-1.0: every character clearly legible.
  - 0.6-0.9: mostly clear but a few ambiguous symbols.
  - 0.3-0.6: significant ambiguity (e.g. "1" vs "l" vs "I", smudged digit).
  - 0.0-0.3: barely readable.

Return STRICT JSON with this shape (no markdown, no commentary):
  {{ "lines": [
    {{ "latex": "<latex line 1>", "confidence": 0.95 }},
    {{ "latex": "<latex line 2>", "confidence": 0.62 }}
  ] }}

If the image is unreadable or contains no math, return {{ "lines": [] }}."""


def _gemini_text(prompt: str) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai not installed; skipping Gemini call")
        return None
    # Default to flash-lite across the board — lower latency, sufficient for
    # short K-12 explanations and hints. Override per call site with
    # GEMINI_MODEL if a heavier model is needed.
    model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        text = (response.text or "").strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 — never crash callers
        logger.warning("Gemini call failed: %s", exc)
        return None


def explain_mistake(
    problem: str, expected: str, submitted: str, locale: Locale = "en"
) -> str | None:
    return _gemini_text(
        _PROMPT_TEMPLATE.format(
            problem=problem,
            expected=expected,
            submitted=submitted,
            lang=_language_directive(locale),
        )
    )


def explain_mistake_detailed(
    problem: str, expected: str, submitted: str, locale: Locale = "en"
) -> str | None:
    """2-3 sentence diagnostic explanation in the requested locale. Falls back
    to None when Gemini is unavailable."""
    return _gemini_text(
        _WHY_WRONG_PROMPT.format(
            problem=problem,
            expected=expected,
            submitted=submitted,
            lang=_language_directive(locale),
        )
    )


_HINT_PROMPT = """You are a math tutor for K-12. The student is solving a problem
step-by-step and is stuck on step {step_num} of {total_steps}.

Problem (LaTeX): {problem}
What the student has written so far (each line = one prior step):
{prior}
Expected next step (LaTeX, do NOT reveal this verbatim): {expected}
Last step the student wrote (LaTeX): {last_line}

Write ONE sentence, 12-22 words, no LaTeX, no markdown, no emoji.

The sentence MUST:
  1. Name the SPECIFIC operation needed next (e.g. "distribute the -3 over (x+2)",
     "combine the like terms 5x and -2x", "add 4 to both sides", "factor out 2x").
  2. Reference at least one concrete symbol, coefficient, or term that appears
     in the problem or the student's last step (e.g. "the 3 in front of (x-1)").
  3. NOT give the expected step or the final answer.
  4. NOT be generic ("look at the operation", "what should you do next") —
     a generic sentence means you failed.

Output only the sentence.

{lang}"""


def generate_hint(
    problem: str,
    expected: str,
    prior_lines: list[str],
    step_num: int,
    total_steps: int,
    locale: Locale = "en",
) -> str | None:
    prior_block = (
        "\n".join(f"  {i + 1}. {line}" for i, line in enumerate(prior_lines))
        if prior_lines
        else "  (no prior steps written yet — this is the first step after the problem)"
    )
    last_line = prior_lines[-1] if prior_lines else problem
    return _gemini_text(
        _HINT_PROMPT.format(
            problem=problem,
            expected=expected,
            prior=prior_block,
            last_line=last_line,
            step_num=step_num,
            total_steps=total_steps,
            lang=_language_directive(locale),
        )
    )


def gemini_is_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


_GENERATE_VARIANT_PROMPT = """You are a math problem generator for K-12.
Given this seed problem and its step-by-step solution, generate {n} variant
problems at difficulty {difficulty_delta} (-1 = easier, 0 = same level, +1 = harder).

Each variant must:
  - Use the SAME type of operation as the seed (e.g. distribution, like-terms).
  - Have DIFFERENT numbers/structure than the seed.
  - Include the full step-by-step solution.

Seed problem (LaTeX): {prompt}
Seed solution steps:
{solution}

Return STRICT JSON (no markdown, no commentary):
{{ "variants": [
  {{ "prompt_latex": "...", "solution_latex": ["step1 latex", "step2 latex", ...] }}
] }}"""


def generate_question_variants(
    seed_prompt: str, seed_solution: list[str], n: int, difficulty_delta: int
) -> list[dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return []
    try:
        from google import genai
    except ImportError:
        return []
    model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    sol = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(seed_solution))
    prompt = _GENERATE_VARIANT_PROMPT.format(
        n=n,
        difficulty_delta=difficulty_delta,
        prompt=seed_prompt,
        solution=sol,
    )
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini generate failed: %s", exc)
        return []
    raw = (response.text or "").strip()
    cleaned = _strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("variant generation: not valid JSON: %r", raw[:200])
        return []
    variants = data.get("variants") if isinstance(data, dict) else None
    if not isinstance(variants, list):
        return []
    out: list[dict] = []
    for v in variants:
        if not isinstance(v, dict):
            continue
        prompt_latex = str(v.get("prompt_latex", "")).strip()
        solution_latex = v.get("solution_latex", [])
        if not prompt_latex or not isinstance(solution_latex, list):
            continue
        steps = [str(s).strip() for s in solution_latex if str(s).strip()]
        if not steps:
            continue
        out.append({"prompt_latex": prompt_latex, "solution_latex": steps})
    return out


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
) -> list[dict]:
    """Run Gemini Vision over a handwritten image. Returns a list of
    {"latex": str, "confidence": float} dicts, one per visual line.
    Returns an empty list if Gemini is unavailable or the response can't be
    parsed — the caller decides how to surface that."""
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

    # OCR doesn't need the full Flash model — Flash-Lite is faster and cheaper
    # and handles short handwritten LaTeX lines well. Override with
    # GEMINI_OCR_MODEL if needed.
    model = os.getenv("GEMINI_OCR_MODEL", "gemini-flash-lite-latest")
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
    out: list[dict] = []
    for entry in lines:
        if isinstance(entry, dict):
            latex = str(entry.get("latex", "")).strip()
            try:
                conf = float(entry.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
        else:
            latex = str(entry).strip()
            conf = 0.5
        if latex:
            out.append({"latex": latex, "confidence": conf})
    return out
