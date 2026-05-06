"""Seed question bank for the Math Practice System.

Each question stores:
  - id: stable identifier
  - prompt_latex: the original problem (LaTeX)
  - solution_latex: list of intermediate-step LaTeX strings, in order.
    The final element is the fully simplified answer.

The solution lines were taken from the sample workbook page
(Q1 in the brief). Q2/Q3 are placeholder problems of similar shape.
"""

from typing import TypedDict


class Question(TypedDict):
    id: str
    prompt_latex: str
    solution_latex: list[str]


QUESTIONS: list[Question] = [
    {
        "id": "q1",
        "prompt_latex": r"x^2 - [2x + \{(x^2 - 1) - (2x^2 + 1)\}]",
        "solution_latex": [
            r"x^2 - \{2x + (x^2 - 1 - 2x^2 - 1)\}",
            r"x^2 - \{2x + (-x^2 - 2)\}",
            r"x^2 - (2x - x^2 - 2)",
            r"x^2 - 2x + x^2 + 2",
            r"2x^2 - 2x + 2",
        ],
    },
    {
        "id": "q2",
        "prompt_latex": r"(3a + 2b) - (a - 4b)",
        "solution_latex": [
            r"3a + 2b - a + 4b",
            r"2a + 6b",
        ],
    },
    {
        "id": "q3",
        "prompt_latex": r"2(x + 3) + 3(2x - 1)",
        "solution_latex": [
            r"2x + 6 + 6x - 3",
            r"8x + 3",
        ],
    },
]


def get_questions() -> list[Question]:
    return QUESTIONS


def get_question(qid: str) -> Question | None:
    return next((q for q in QUESTIONS if q["id"] == qid), None)
