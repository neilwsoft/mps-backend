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

from migrations import run_migrations
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
        run_migrations(conn)

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
                   sl.partial_score, sl.time_spent_ms, sl.source, sl.ocr_confidence,
                   sl.override_correct, sl.override_reason, sl.override_at,
                   ou.name AS override_by_name,
                   eq.position AS question_position,
                   eq.prompt_latex AS question_prompt
            FROM submission_lines sl
            JOIN exam_questions eq ON eq.id = sl.question_id
            LEFT JOIN users ou ON ou.id = sl.override_by
            WHERE sl.submission_id = ?
            ORDER BY eq.position, sl.line_index, sl.id
            """,
            (submission_id,),
        ).fetchall()
        hints = conn.execute(
            """SELECT question_id, line_index, hint_text, created_at
               FROM hints_used WHERE submission_id = ?
               ORDER BY created_at""",
            (submission_id,),
        ).fetchall()
    return {
        **dict(sub),
        "lines": [dict(r) for r in lines],
        "hints": [dict(r) for r in hints],
    }


def record_submission_line(
    submission_id: int,
    question_id: int,
    line_index: int,
    submitted_latex: str,
    correct: bool,
    explanation: str | None,
    partial_score: float = 0.0,
    time_spent_ms: int | None = None,
    source: str = "typed",
    ocr_confidence: float | None = None,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO submission_lines "
            "(submission_id, question_id, line_index, submitted_latex, correct, "
            " explanation, partial_score, time_spent_ms, source, ocr_confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                submission_id,
                question_id,
                line_index,
                submitted_latex,
                int(correct),
                explanation,
                float(partial_score),
                time_spent_ms,
                source,
                ocr_confidence,
            ),
        )
        return int(cur.lastrowid)


def _compute_score(conn: sqlite3.Connection, submission_id: int) -> tuple[int, int]:
    """Return (score, total_expected). Score honors overrides and partial credit:
      - if override_correct is set, that takes precedence (1 → full point, 0 → no points)
      - else use partial_score (0.0..1.0)
    Hints used cost 1 point each (capped at score). Score is rounded to nearest int.
    """
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

    raw = conn.execute(
        """SELECT
             COALESCE(
               CASE WHEN override_correct IS NULL THEN NULL ELSE override_correct * 1.0 END,
               partial_score
             ) AS pts
           FROM submission_lines
           WHERE submission_id = ?""",
        (submission_id,),
    ).fetchall()
    raw_score = sum(float(r["pts"]) for r in raw)

    hint_count = conn.execute(
        "SELECT COUNT(*) AS n FROM hints_used WHERE submission_id = ?",
        (submission_id,),
    ).fetchone()["n"]

    final = max(0, round(raw_score - int(hint_count)))
    return min(final, total_expected), total_expected


def recompute_score(submission_id: int) -> dict:
    """Recompute score for a (possibly already finalized) submission. Used
    after grade overrides."""
    with connect() as conn:
        score, total = _compute_score(conn, submission_id)
        conn.execute(
            "UPDATE submissions SET score = ?, total = ? WHERE id = ?",
            (score, total, submission_id),
        )
    return {"score": score, "total": total}


def finalize_submission(submission_id: int) -> dict:
    """Compute score from recorded lines and stamp submitted_at."""
    with connect() as conn:
        score, total_expected = _compute_score(conn, submission_id)
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


def record_hint(
    submission_id: int, question_id: int, line_index: int, hint_text: str
) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO hints_used (submission_id, question_id, line_index, hint_text) "
            "VALUES (?, ?, ?, ?)",
            (submission_id, question_id, line_index, hint_text),
        )
        return int(cur.lastrowid)


def count_hints(submission_id: int, question_id: int, line_index: int) -> int:
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM hints_used "
            "WHERE submission_id = ? AND question_id = ? AND line_index = ?",
            (submission_id, question_id, line_index),
        ).fetchone()
    return int(row["n"])


def get_prior_lines(submission_id: int, question_id: int, line_index: int) -> list[str]:
    """Lines the student already submitted for this question, in order."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT submitted_latex FROM submission_lines
               WHERE submission_id = ? AND question_id = ? AND line_index < ?
               ORDER BY line_index, id""",
            (submission_id, question_id, line_index),
        ).fetchall()
    return [r["submitted_latex"] for r in rows]


# ----- scratchpad ---------------------------------------------------------


def upsert_scratchpad(submission_id: int, question_id: int, content: str) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT INTO scratchpad (submission_id, question_id, content)
               VALUES (?, ?, ?)
               ON CONFLICT(submission_id, question_id)
               DO UPDATE SET content = excluded.content, updated_at = datetime('now')""",
            (submission_id, question_id, content),
        )


def get_scratchpad(submission_id: int, question_id: int) -> str:
    with connect() as conn:
        row = conn.execute(
            "SELECT content FROM scratchpad WHERE submission_id = ? AND question_id = ?",
            (submission_id, question_id),
        ).fetchone()
    return row["content"] if row else ""


# ----- grade overrides ----------------------------------------------------


def override_submission_line(
    line_id: int, override_correct: bool, reason: str, admin_user_id: int
) -> dict | None:
    with connect() as conn:
        cur = conn.execute(
            """UPDATE submission_lines
               SET override_correct = ?, override_reason = ?,
                   override_by = ?, override_at = datetime('now')
               WHERE id = ?""",
            (int(override_correct), reason, admin_user_id, line_id),
        )
        if cur.rowcount == 0:
            return None
        row = conn.execute(
            "SELECT submission_id FROM submission_lines WHERE id = ?", (line_id,)
        ).fetchone()
    if row is None:
        return None
    return recompute_score(row["submission_id"])


# ----- exam clone ---------------------------------------------------------


def clone_exam(source_exam_id: int, created_by: int, title_suffix: str = " (copy)") -> int | None:
    with connect() as conn:
        src = conn.execute(
            "SELECT title, description FROM exams WHERE id = ?",
            (source_exam_id,),
        ).fetchone()
        if src is None:
            return None
        cur = conn.execute(
            "INSERT INTO exams (title, description, created_by, cloned_from_id) "
            "VALUES (?, ?, ?, ?)",
            (src["title"] + title_suffix, src["description"], created_by, source_exam_id),
        )
        new_exam_id = int(cur.lastrowid)
        for q in conn.execute(
            "SELECT position, prompt_latex, solution_latex FROM exam_questions "
            "WHERE exam_id = ? ORDER BY position",
            (source_exam_id,),
        ).fetchall():
            conn.execute(
                "INSERT INTO exam_questions (exam_id, position, prompt_latex, solution_latex) "
                "VALUES (?, ?, ?, ?)",
                (new_exam_id, q["position"], q["prompt_latex"], q["solution_latex"]),
            )
    return new_exam_id


# ----- topics -------------------------------------------------------------


def list_topics() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT id, name FROM topics ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def upsert_topic(name: str) -> int:
    name = name.strip()
    if not name:
        raise ValueError("topic name required")
    with connect() as conn:
        existing = conn.execute("SELECT id FROM topics WHERE name = ?", (name,)).fetchone()
        if existing:
            return int(existing["id"])
        cur = conn.execute("INSERT INTO topics (name) VALUES (?)", (name,))
        return int(cur.lastrowid)


def set_question_topics(question_id: int, topic_names: list[str]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM question_topics WHERE question_id = ?", (question_id,))
        for raw in topic_names:
            name = raw.strip()
            if not name:
                continue
            existing = conn.execute("SELECT id FROM topics WHERE name = ?", (name,)).fetchone()
            if existing:
                tid = existing["id"]
            else:
                cur = conn.execute("INSERT INTO topics (name) VALUES (?)", (name,))
                tid = cur.lastrowid
            conn.execute(
                "INSERT OR IGNORE INTO question_topics (question_id, topic_id) VALUES (?, ?)",
                (question_id, tid),
            )


def get_question_topics(question_id: int) -> list[str]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT t.name FROM topics t
               JOIN question_topics qt ON qt.topic_id = t.id
               WHERE qt.question_id = ? ORDER BY t.name""",
            (question_id,),
        ).fetchall()
    return [r["name"] for r in rows]


# ----- analytics ----------------------------------------------------------


def question_metrics(exam_id: int) -> list[dict]:
    """Per-question metrics: pct correct, mean time, attempt count, discrimination."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT eq.id AS question_id, eq.position, eq.prompt_latex,
                      COUNT(sl.id) AS attempts,
                      SUM(sl.correct) AS correct_count,
                      AVG(sl.time_spent_ms) AS avg_ms
               FROM exam_questions eq
               LEFT JOIN submission_lines sl ON sl.question_id = eq.id
               WHERE eq.exam_id = ?
               GROUP BY eq.id
               ORDER BY eq.position""",
            (exam_id,),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        attempts = int(r["attempts"] or 0)
        correct = int(r["correct_count"] or 0)
        out.append(
            {
                "question_id": r["question_id"],
                "position": r["position"],
                "prompt_latex": r["prompt_latex"],
                "attempts": attempts,
                "pct_correct": (correct / attempts) if attempts else 0.0,
                "avg_seconds": ((r["avg_ms"] or 0) / 1000.0) if r["avg_ms"] else None,
            }
        )
    return out


def student_analytics(user_id: int) -> dict:
    """Rich per-student analytics: summary, per-exam, per-topic, per-question,
    hints, total time spent. Used by the admin student-detail page."""
    with connect() as conn:
        student = conn.execute(
            "SELECT id, name, email, created_at FROM users WHERE id = ? AND role = 'student'",
            (user_id,),
        ).fetchone()
        if student is None:
            return {}
        per_exam = conn.execute(
            """SELECT s.id AS submission_id, s.exam_id, e.title AS exam_title,
                      s.started_at, s.submitted_at, s.score, s.total,
                      (SELECT COUNT(*) FROM submission_lines sl
                       WHERE sl.submission_id = s.id) AS line_count,
                      (SELECT SUM(sl.time_spent_ms) FROM submission_lines sl
                       WHERE sl.submission_id = s.id) AS total_time_ms,
                      (SELECT COUNT(*) FROM hints_used h
                       WHERE h.submission_id = s.id) AS hints_used
               FROM submissions s
               JOIN exams e ON e.id = s.exam_id
               WHERE s.user_id = ?
               ORDER BY s.started_at""",
            (user_id,),
        ).fetchall()
        per_topic = conn.execute(
            """SELECT t.name AS topic,
                      COUNT(sl.id) AS attempts,
                      SUM(sl.correct) AS correct_count
               FROM topics t
               JOIN question_topics qt ON qt.topic_id = t.id
               JOIN submission_lines sl ON sl.question_id = qt.question_id
               JOIN submissions s ON s.id = sl.submission_id
               WHERE s.user_id = ?
               GROUP BY t.id
               ORDER BY t.name""",
            (user_id,),
        ).fetchall()
        per_question = conn.execute(
            """SELECT eq.id AS question_id, eq.prompt_latex,
                      e.title AS exam_title,
                      COUNT(sl.id) AS attempts,
                      SUM(sl.correct) AS correct_count,
                      AVG(sl.time_spent_ms) AS avg_ms
               FROM exam_questions eq
               JOIN submission_lines sl ON sl.question_id = eq.id
               JOIN submissions s ON s.id = sl.submission_id
               JOIN exams e ON e.id = eq.exam_id
               WHERE s.user_id = ?
               GROUP BY eq.id
               ORDER BY e.title, eq.position""",
            (user_id,),
        ).fetchall()

    exams_out: list[dict] = []
    for r in per_exam:
        total = int(r["total"] or 0)
        score = int(r["score"] or 0)
        exams_out.append(
            {
                "submission_id": r["submission_id"],
                "exam_id": r["exam_id"],
                "exam_title": r["exam_title"],
                "started_at": r["started_at"],
                "submitted_at": r["submitted_at"],
                "score": score,
                "total": total,
                "accuracy": (score / total) if total else None,
                "line_count": int(r["line_count"] or 0),
                "total_time_ms": int(r["total_time_ms"] or 0),
                "hints_used": int(r["hints_used"] or 0),
            }
        )
    topics_out = [
        {
            "topic": r["topic"],
            "attempts": int(r["attempts"]),
            "correct": int(r["correct_count"] or 0),
            "accuracy": (int(r["correct_count"] or 0) / int(r["attempts"]))
            if r["attempts"]
            else 0.0,
        }
        for r in per_topic
    ]
    questions_out = [
        {
            "question_id": r["question_id"],
            "prompt_latex": r["prompt_latex"],
            "exam_title": r["exam_title"],
            "attempts": int(r["attempts"]),
            "correct": int(r["correct_count"] or 0),
            "accuracy": (int(r["correct_count"] or 0) / int(r["attempts"]))
            if r["attempts"]
            else 0.0,
            "avg_seconds": (r["avg_ms"] / 1000.0) if r["avg_ms"] else None,
        }
        for r in per_question
    ]
    total_attempts = sum(q["attempts"] for q in questions_out)
    total_correct = sum(q["correct"] for q in questions_out)
    total_time = sum(e["total_time_ms"] for e in exams_out)
    total_hints = sum(e["hints_used"] for e in exams_out)
    return {
        "student": dict(student),
        "summary": {
            "exams_attempted": len(exams_out),
            "exams_finalized": sum(1 for e in exams_out if e["submitted_at"]),
            "total_attempts": total_attempts,
            "total_correct": total_correct,
            "accuracy": (total_correct / total_attempts) if total_attempts else 0.0,
            "total_time_ms": total_time,
            "hints_used": total_hints,
        },
        "exams": exams_out,
        "topics": topics_out,
        "questions": questions_out,
    }


def topic_mastery() -> list[dict]:
    """Per (student, topic) accuracy across all submissions."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT u.id AS student_id, u.name AS student_name,
                      t.id AS topic_id, t.name AS topic,
                      COUNT(sl.id) AS attempts,
                      SUM(sl.correct) AS correct_count
               FROM users u
               JOIN submissions s ON s.user_id = u.id
               JOIN submission_lines sl ON sl.submission_id = s.id
               JOIN question_topics qt ON qt.question_id = sl.question_id
               JOIN topics t ON t.id = qt.topic_id
               WHERE u.role = 'student'
               GROUP BY u.id, t.id
               ORDER BY u.name, t.name"""
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        attempts = int(r["attempts"])
        correct = int(r["correct_count"] or 0)
        out.append(
            {
                "student_id": r["student_id"],
                "student_name": r["student_name"],
                "topic": r["topic"],
                "attempts": attempts,
                "accuracy": (correct / attempts) if attempts else 0.0,
            }
        )
    return out


# ----- legacy --------------------------------------------------------------


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
