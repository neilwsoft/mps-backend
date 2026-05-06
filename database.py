"""SQLite layer for the Math Practice System.

Tables
------
users                 — students and admins (auth)
exams                 — exam metadata (created by an admin)
exam_questions        — questions belonging to an exam (prompt + ordered
                        list of model-solution lines, JSON-encoded)
submissions           — one row per student attempt at an exam
submission_lines      — every line a student submitted, with verdict +
                        explanation (audit trail)

Init behavior
-------------
``init_db`` creates the schema, seeds an admin user, and seeds a single
"Demo Algebra" exam from the legacy QUESTIONS bank if no exams exist.
Schema is forward-compatible: legacy ``questions``/``attempts`` tables
from earlier revisions are dropped automatically.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from questions import QUESTIONS

DB_PATH = os.getenv("DATABASE_PATH", "./mps.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    hashed_password TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('student', 'admin')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS exams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_by INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (created_by) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS exam_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exam_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    prompt_latex TEXT NOT NULL,
    solution_latex TEXT NOT NULL,  -- JSON array of strings
    FOREIGN KEY (exam_id) REFERENCES exams(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exam_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    submitted_at TEXT,
    score INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (exam_id) REFERENCES exams(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS submission_lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL,
    question_id INTEGER NOT NULL,
    line_index INTEGER NOT NULL,
    submitted_latex TEXT NOT NULL,
    correct INTEGER NOT NULL,
    explanation TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (submission_id) REFERENCES submissions(id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES exam_questions(id) ON DELETE CASCADE
);
"""

# Legacy tables (pre-auth schema). Dropped on first init so the dev DB
# isn't carrying around unused state.
LEGACY_DROPS = (
    "DROP TABLE IF EXISTS attempts",
    "DROP TABLE IF EXISTS questions",
)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    from auth import hash_password

    admin_email = os.getenv("ADMIN_EMAIL", "admin@email.com")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    admin_name = os.getenv("ADMIN_NAME", "Admin")

    with connect() as conn:
        for stmt in LEGACY_DROPS:
            conn.execute(stmt)
        conn.executescript(SCHEMA)

        admin = conn.execute(
            "SELECT id FROM users WHERE email = ?", (admin_email,)
        ).fetchone()
        if admin is None:
            cur = conn.execute(
                "INSERT INTO users (email, name, hashed_password, role) VALUES (?, ?, ?, 'admin')",
                (admin_email, admin_name, hash_password(admin_password)),
            )
            admin_id = cur.lastrowid
        else:
            admin_id = admin["id"]

        n_exams = conn.execute("SELECT COUNT(*) AS n FROM exams").fetchone()["n"]
        if n_exams == 0 and admin_id is not None:
            cur = conn.execute(
                "INSERT INTO exams (title, description, created_by) VALUES (?, ?, ?)",
                (
                    "Demo Algebra",
                    "A short warm-up: bracket expansion and like-term collection.",
                    admin_id,
                ),
            )
            exam_id = cur.lastrowid
            for pos, q in enumerate(QUESTIONS):
                conn.execute(
                    "INSERT INTO exam_questions (exam_id, position, prompt_latex, solution_latex) "
                    "VALUES (?, ?, ?, ?)",
                    (exam_id, pos, q["prompt_latex"], json.dumps(q["solution_latex"])),
                )


# ----- users --------------------------------------------------------------


def get_user_by_email(email: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, email, name, hashed_password, role, created_at "
            "FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    return dict(row) if row else None


def get_user(user_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, email, name, role, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def create_student(email: str, name: str, hashed_password: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, name, hashed_password, role) VALUES (?, ?, ?, 'student')",
            (email, name, hashed_password),
        )
        return int(cur.lastrowid)


def list_students() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
              u.id, u.email, u.name, u.created_at,
              COUNT(DISTINCT s.id) AS submission_count,
              COALESCE(SUM(s.score), 0) AS total_score,
              COALESCE(SUM(s.total), 0) AS total_lines
            FROM users u
            LEFT JOIN submissions s ON s.user_id = u.id AND s.submitted_at IS NOT NULL
            WHERE u.role = 'student'
            GROUP BY u.id
            ORDER BY u.created_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


# ----- exams --------------------------------------------------------------


def list_exams() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT e.id, e.title, e.description, e.created_at,
                   COUNT(eq.id) AS question_count
            FROM exams e
            LEFT JOIN exam_questions eq ON eq.exam_id = e.id
            GROUP BY e.id
            ORDER BY e.created_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_exam(exam_id: int) -> dict | None:
    with connect() as conn:
        exam = conn.execute(
            "SELECT id, title, description, created_at FROM exams WHERE id = ?",
            (exam_id,),
        ).fetchone()
        if exam is None:
            return None
        questions = conn.execute(
            "SELECT id, position, prompt_latex, solution_latex "
            "FROM exam_questions WHERE exam_id = ? ORDER BY position",
            (exam_id,),
        ).fetchall()
    return {
        **dict(exam),
        "questions": [
            {
                "id": q["id"],
                "position": q["position"],
                "prompt_latex": q["prompt_latex"],
                "solution_latex": json.loads(q["solution_latex"]),
            }
            for q in questions
        ],
    }


def get_exam_question(question_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, exam_id, position, prompt_latex, solution_latex "
            "FROM exam_questions WHERE id = ?",
            (question_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "exam_id": row["exam_id"],
        "position": row["position"],
        "prompt_latex": row["prompt_latex"],
        "solution_latex": json.loads(row["solution_latex"]),
    }


def create_exam(
    title: str,
    description: str,
    created_by: int,
    questions: list[dict],
) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO exams (title, description, created_by) VALUES (?, ?, ?)",
            (title, description, created_by),
        )
        exam_id = int(cur.lastrowid)
        for pos, q in enumerate(questions):
            conn.execute(
                "INSERT INTO exam_questions (exam_id, position, prompt_latex, solution_latex) "
                "VALUES (?, ?, ?, ?)",
                (
                    exam_id,
                    pos,
                    q["prompt_latex"],
                    json.dumps(q["solution_latex"]),
                ),
            )
        return exam_id


# ----- submissions --------------------------------------------------------


def create_submission(exam_id: int, user_id: int) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO submissions (exam_id, user_id) VALUES (?, ?)",
            (exam_id, user_id),
        )
        return int(cur.lastrowid)


def get_submission(submission_id: int) -> dict | None:
    with connect() as conn:
        sub = conn.execute(
            """
            SELECT s.id, s.exam_id, s.user_id, s.started_at, s.submitted_at,
                   s.score, s.total,
                   e.title AS exam_title, e.description AS exam_description,
                   u.email AS student_email, u.name AS student_name
            FROM submissions s
            JOIN exams e ON e.id = s.exam_id
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ?
            """,
            (submission_id,),
        ).fetchone()
        if sub is None:
            return None
        lines = conn.execute(
            """
            SELECT sl.id, sl.question_id, sl.line_index, sl.submitted_latex,
                   sl.correct, sl.explanation, sl.created_at,
                   eq.position AS question_position,
                   eq.prompt_latex AS question_prompt
            FROM submission_lines sl
            JOIN exam_questions eq ON eq.id = sl.question_id
            WHERE sl.submission_id = ?
            ORDER BY eq.position, sl.line_index, sl.id
            """,
            (submission_id,),
        ).fetchall()
    return {**dict(sub), "lines": [dict(r) for r in lines]}


def record_submission_line(
    submission_id: int,
    question_id: int,
    line_index: int,
    submitted_latex: str,
    correct: bool,
    explanation: str | None,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO submission_lines "
            "(submission_id, question_id, line_index, submitted_latex, correct, explanation) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                submission_id,
                question_id,
                line_index,
                submitted_latex,
                int(correct),
                explanation,
            ),
        )
        return int(cur.lastrowid)


def finalize_submission(submission_id: int) -> dict:
    """Compute score from recorded lines and stamp submitted_at.

    Score = number of correct lines. Total = expected total lines across
    all exam questions (so partial submissions are penalised — the
    student got 0 for the lines they didn't attempt).
    """
    with connect() as conn:
        sub = conn.execute(
            "SELECT exam_id FROM submissions WHERE id = ?", (submission_id,)
        ).fetchone()
        if sub is None:
            raise ValueError(f"submission {submission_id} not found")
        questions = conn.execute(
            "SELECT solution_latex FROM exam_questions WHERE exam_id = ?",
            (sub["exam_id"],),
        ).fetchall()
        total_expected = sum(len(json.loads(q["solution_latex"])) for q in questions)
        correct_row = conn.execute(
            "SELECT COUNT(*) AS n FROM submission_lines "
            "WHERE submission_id = ? AND correct = 1",
            (submission_id,),
        ).fetchone()
        score = int(correct_row["n"])
        conn.execute(
            "UPDATE submissions SET submitted_at = datetime('now'), score = ?, total = ? "
            "WHERE id = ?",
            (score, total_expected, submission_id),
        )
    return {"score": score, "total": total_expected}


def list_submissions_for_user(user_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.exam_id, s.started_at, s.submitted_at, s.score, s.total,
                   e.title AS exam_title
            FROM submissions s
            JOIN exams e ON e.id = s.exam_id
            WHERE s.user_id = ?
            ORDER BY s.started_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_all_submissions() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.exam_id, s.user_id, s.started_at, s.submitted_at,
                   s.score, s.total,
                   e.title AS exam_title,
                   u.email AS student_email, u.name AS student_name
            FROM submissions s
            JOIN exams e ON e.id = s.exam_id
            JOIN users u ON u.id = s.user_id
            ORDER BY s.started_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def list_submissions_for_exam(exam_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.user_id, s.started_at, s.submitted_at, s.score, s.total,
                   u.email AS student_email, u.name AS student_name
            FROM submissions s
            JOIN users u ON u.id = s.user_id
            WHERE s.exam_id = ?
            ORDER BY s.started_at DESC
            """,
            (exam_id,),
        ).fetchall()
    return [dict(r) for r in rows]
