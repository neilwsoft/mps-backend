"""Idempotent demo-data seeder.

Run with: ``python seed_demo.py``

What it does:
  1. Tags the existing 3 demo questions with topics (Brackets, Distribution,
     Like terms, Sign manipulation).
  2. Creates a small roster of demo students (skipped if already present).
  3. Generates a spread of finalized submissions per student with realistic
     line patterns (some correct, some wrong, some partial credit, varied
     time spent, occasional hint use). Skips a student if they already
     have submissions.

Re-running is safe — every step checks for prior state.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timedelta

import auth
import database


_DEMO_STUDENTS = [
    ("amelia.rivera@school.test", "Amelia Rivera"),
    ("brandon.kim@school.test", "Brandon Kim"),
    ("carla.singh@school.test", "Carla Singh"),
    ("david.osei@school.test", "David Osei"),
    ("evelyn.tanaka@school.test", "Evelyn Tanaka"),
]

_QUESTION_TOPICS = {
    # exam_questions.position -> list[topic name]
    0: ["Brackets", "Sign manipulation", "Like terms"],
    1: ["Distribution", "Sign manipulation", "Like terms"],
    2: ["Distribution", "Like terms"],
}


def _ensure_topics(conn: sqlite3.Connection) -> None:
    questions = conn.execute(
        "SELECT id, position FROM exam_questions ORDER BY position"
    ).fetchall()
    for q in questions:
        for topic_name in _QUESTION_TOPICS.get(q["position"], []):
            existing = conn.execute(
                "SELECT id FROM topics WHERE name = ?", (topic_name,)
            ).fetchone()
            if existing:
                tid = existing["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO topics (name) VALUES (?)", (topic_name,)
                )
                tid = cur.lastrowid
            conn.execute(
                "INSERT OR IGNORE INTO question_topics (question_id, topic_id) "
                "VALUES (?, ?)",
                (q["id"], tid),
            )


def _ensure_students(conn: sqlite3.Connection) -> list[int]:
    ids: list[int] = []
    for email, name in _DEMO_STUDENTS:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing:
            ids.append(int(existing["id"]))
            continue
        cur = conn.execute(
            "INSERT INTO users (email, name, hashed_password, role) "
            "VALUES (?, ?, ?, 'student')",
            (email, name, auth.hash_password("demo1234")),
        )
        ids.append(int(cur.lastrowid))
    return ids


def _has_demo_submissions(conn: sqlite3.Connection, user_id: int) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM submissions WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return int(row["n"]) > 0


def _seed_submissions(conn: sqlite3.Connection, student_ids: list[int]) -> None:
    rng = random.Random(42)  # reproducible

    exams = conn.execute("SELECT id FROM exams ORDER BY id").fetchall()
    if not exams:
        return
    exam_id = exams[0]["id"]
    questions = conn.execute(
        "SELECT id, position, prompt_latex, solution_latex "
        "FROM exam_questions WHERE exam_id = ? ORDER BY position",
        (exam_id,),
    ).fetchall()

    parsed_qs = [
        {
            "id": q["id"],
            "position": q["position"],
            "prompt_latex": q["prompt_latex"],
            "solution_latex": json.loads(q["solution_latex"]),
        }
        for q in questions
    ]

    # Per-student "skill profile" — chance of getting a line right.
    # Lower = struggles more; higher = stronger student. Spread realistically.
    skills = [0.45, 0.55, 0.7, 0.82, 0.92]

    base_started = datetime(2026, 4, 5, 9, 30)
    for s_idx, sid in enumerate(student_ids):
        if _has_demo_submissions(conn, sid):
            continue
        skill = skills[s_idx % len(skills)]
        # 2-3 submissions per student, spaced over a couple of weeks.
        n_subs = rng.randint(2, 3)
        for k in range(n_subs):
            started = base_started + timedelta(
                days=s_idx * 2 + k * 4,
                minutes=rng.randint(0, 90),
            )
            cur = conn.execute(
                "INSERT INTO submissions (exam_id, user_id, started_at) "
                "VALUES (?, ?, ?)",
                (exam_id, sid, started.strftime("%Y-%m-%d %H:%M:%S")),
            )
            sub_id = int(cur.lastrowid)

            # Walk through every question, every line, with skill-based correctness.
            hints_used = 0
            for q in parsed_qs:
                for li, expected in enumerate(q["solution_latex"]):
                    correct_roll = rng.random() < skill
                    # Some wrong lines are value-equivalent (partial credit).
                    if correct_roll:
                        submitted_latex = expected
                        correct = 1
                        partial = 1.0
                        explanation = None
                    else:
                        # Inject a "wrong-but-close" or "wrong" answer.
                        if rng.random() < 0.4:
                            submitted_latex = f"{expected} + 0"
                            partial = 0.5
                        else:
                            submitted_latex = f"{expected} + 1"
                            partial = 0.0
                        correct = 0
                        explanation = (
                            "Re-check the coefficient on the leading term — "
                            "the constant doesn't match the expected step."
                        )

                    # Time per line: ~12s + skill-modulated jitter.
                    base_secs = 8 + (1 - skill) * 30
                    line_secs = max(3, rng.gauss(base_secs, 4))
                    time_ms = int(line_secs * 1000)

                    conn.execute(
                        """INSERT INTO submission_lines
                           (submission_id, question_id, line_index, submitted_latex,
                            correct, explanation, partial_score, time_spent_ms,
                            source, ocr_confidence)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'typed', NULL)""",
                        (
                            sub_id,
                            q["id"],
                            li,
                            submitted_latex,
                            correct,
                            explanation,
                            partial,
                            time_ms,
                        ),
                    )

                    # Occasionally a struggling student asks for a hint.
                    if not correct_roll and rng.random() < 0.2:
                        conn.execute(
                            """INSERT INTO hints_used
                               (submission_id, question_id, line_index, hint_text)
                               VALUES (?, ?, ?, ?)""",
                            (
                                sub_id,
                                q["id"],
                                li,
                                "Distribute the coefficient over every term "
                                "inside the parentheses.",
                            ),
                        )
                        hints_used += 1

            # Compute and stamp finalize.
            score, total = database._compute_score(conn, sub_id)
            submitted_at = (started + timedelta(minutes=rng.randint(8, 25))).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            conn.execute(
                "UPDATE submissions SET submitted_at = ?, score = ?, total = ? "
                "WHERE id = ?",
                (submitted_at, score, total, sub_id),
            )


def seed() -> None:
    database.init_db()  # ensure schema + migrations
    with database.connect() as conn:
        _ensure_topics(conn)
        student_ids = _ensure_students(conn)
        _seed_submissions(conn, student_ids)
    print("✓ demo data seeded (topics, students, submissions)")


if __name__ == "__main__":
    seed()
