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
    partial_score: float = 0.0  # 0.0 .. 1.0; 1.0 when correct, 0.5 for value-equivalent, 0.0 wrong


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


_BASELINE_MESSAGES = {
    "en": {
        "parse_error": "Could not parse your line — check brackets, operators, and ^ exponents.",
        "internal_error": "Internal error parsing the model solution.",
        "value_correct": "Value is correct but the form doesn't match this step — please write it as: {expected}",
        "equals_problem": "Line still equals the original problem — perform the expected step: {expected}",
        "arithmetic_mismatch": "Arithmetic mismatch — your line simplifies to {got}, but this step should simplify to {want}.",
    },
    "ko": {
        "parse_error": "줄을 해석할 수 없습니다 — 괄호·연산자·지수(^) 표기를 확인해 주세요.",
        "internal_error": "모범 풀이를 해석하는 중 내부 오류가 발생했습니다.",
        "value_correct": "값은 맞지만 이번 단계의 형태와 다릅니다 — 다음과 같이 적어 주세요: {expected}",
        "equals_problem": "줄이 원래 문제와 같습니다 — 다음 단계를 실행해 주세요: {expected}",
        "arithmetic_mismatch": "계산이 맞지 않습니다 — 적은 줄은 {got}(으)로 간단해지지만, 이 단계는 {want}(으)로 간단해져야 합니다.",
    },
}


def _baseline_explanation(
    submitted_latex: str,
    expected_latex: str,
    problem_latex: str,
    locale: str = "en",
) -> str:
    """Deterministic, sympy-driven fallback when Gemini isn't available."""
    msgs = _BASELINE_MESSAGES.get(locale, _BASELINE_MESSAGES["en"])
    try:
        submitted = _parse(submitted_latex)
    except Exception:
        return msgs["parse_error"]

    try:
        expected = _parse(expected_latex)
    except Exception:
        return msgs["internal_error"]

    if _equivalent(submitted, expected):
        return msgs["value_correct"].format(expected=expected_latex)
    try:
        problem = _parse(problem_latex)
        if _equivalent(submitted, problem):
            return msgs["equals_problem"].format(expected=expected_latex)
    except Exception:
        pass

    return msgs["arithmetic_mismatch"].format(
        got=sympy.simplify(submitted), want=sympy.simplify(expected)
    )


def _value_equivalent(submitted_latex: str, expected_latex: str) -> bool:
    """True when the parsed expressions simplify to the same value, even if the
    rendered LaTeX differs (e.g. `2x + 4` vs `4 + 2x`)."""
    try:
        a = _parse(submitted_latex)
        b = _parse(expected_latex)
    except Exception:
        return False
    return _equivalent(a, b)


def verify_line(
    submitted_latex: str,
    expected_latex: str,
    problem_latex: str,
    line_index: int,
    total_lines: int,
    locale: str = "en",
) -> VerifyResult:
    is_final = line_index == total_lines - 1

    if _normalize(submitted_latex) == _normalize(expected_latex):
        return VerifyResult(
            correct=True, explanation=None, is_final=is_final, partial_score=1.0
        )

    ai_msg = gemini_client.explain_mistake_detailed(
        problem=problem_latex,
        expected=expected_latex,
        submitted=submitted_latex,
        locale=locale,
    )
    explanation = ai_msg or _baseline_explanation(
        submitted_latex, expected_latex, problem_latex, locale=locale
    )

    partial = 0.5 if _value_equivalent(submitted_latex, expected_latex) else 0.0
    return VerifyResult(
        correct=False,
        explanation=explanation,
        is_final=False,
        partial_score=partial,
    )
