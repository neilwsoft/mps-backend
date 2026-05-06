"""Line-by-line verification of a student's solution against the model.

Verdict layer (deterministic):
    Strict structural match via normalized LaTeX strings. Pure-local, no AI.

Explanation layer (only used when verdict is wrong):
    1. Sympy classifies the error type (alt-form vs. equals-original vs.
       arithmetic mismatch) and produces a baseline sentence.
    2. If GEMINI_API_KEY is set, Gemini is asked for a kid-friendlier
       one-sentence explanation, which replaces the sympy sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sympy
from sympy.parsing.latex import parse_latex

import gemini as gemini_client


@dataclass
class VerifyResult:
    correct: bool
    explanation: str | None
    is_final: bool


_NORM_REPLACEMENTS = (
    (r"\\{", "{"),
    (r"\\}", "}"),
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\cdot", "*"),
    (r"\\times", "*"),
)


def _normalize(latex: str) -> str:
    s = latex.strip()
    if s.startswith("="):
        s = s[1:].strip()
    for pat, repl in _NORM_REPLACEMENTS:
        s = re.sub(pat, repl, s)
    s = re.sub(r"\^\{(\w)\}", r"^\1", s)
    s = re.sub(r"\s+", "", s)
    return s


def _parse(latex: str) -> sympy.Expr:
    cleaned = latex.strip()
    if cleaned.startswith("="):
        cleaned = cleaned[1:].strip()
    cleaned = (
        cleaned.replace(r"\{", "(")
        .replace(r"\}", ")")
        .replace(r"\left", "")
        .replace(r"\right", "")
        .replace("{", "(")
        .replace("}", ")")
        .replace("[", "(")
        .replace("]", ")")
    )
    return parse_latex(cleaned)


def _equivalent(a: sympy.Expr, b: sympy.Expr) -> bool:
    try:
        return sympy.simplify(a - b) == 0
    except Exception:
        return False


def _baseline_explanation(
    submitted_latex: str, expected_latex: str, problem_latex: str
) -> str:
    """Deterministic, sympy-driven fallback when Gemini isn't available."""
    try:
        submitted = _parse(submitted_latex)
    except Exception:
        return "Could not parse your line — check brackets, operators, and ^ exponents."

    try:
        expected = _parse(expected_latex)
    except Exception:
        return "Internal error parsing the model solution."

    if _equivalent(submitted, expected):
        return (
            "Value is correct but the form doesn't match this step — "
            f"please write it as: {expected_latex}"
        )
    try:
        problem = _parse(problem_latex)
        if _equivalent(submitted, problem):
            return (
                "Line still equals the original problem — perform the expected step: "
                f"{expected_latex}"
            )
    except Exception:
        pass

    return (
        f"Arithmetic mismatch — your line simplifies to {sympy.simplify(submitted)}, "
        f"but this step should simplify to {sympy.simplify(expected)}."
    )


def verify_line(
    submitted_latex: str,
    expected_latex: str,
    problem_latex: str,
    line_index: int,
    total_lines: int,
) -> VerifyResult:
    is_final = line_index == total_lines - 1

    if _normalize(submitted_latex) == _normalize(expected_latex):
        return VerifyResult(correct=True, explanation=None, is_final=is_final)

    ai_msg = gemini_client.explain_mistake(
        problem=problem_latex, expected=expected_latex, submitted=submitted_latex
    )
    explanation = ai_msg or _baseline_explanation(
        submitted_latex, expected_latex, problem_latex
    )
    return VerifyResult(correct=False, explanation=explanation, is_final=False)
