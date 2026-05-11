"""Idempotent schema migrations.

Each migration is a (name, list[sql_statement]) pair. Applied names are
recorded in ``schema_migrations``; rerunning is a no-op. ALTER TABLE ADD
COLUMN is non-idempotent at the SQLite level, so we swallow "duplicate
column" errors so the same migration is safe to retry on a partially
applied DB.
"""

from __future__ import annotations

import sqlite3

CREATE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  name TEXT PRIMARY KEY,
  applied_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

MIGRATIONS: list[tuple[str, list[str]]] = [
    (
        "001_exam_modes",
        [
            "ALTER TABLE exams ADD COLUMN mode TEXT NOT NULL DEFAULT 'exam'",
            "ALTER TABLE exams ADD COLUMN time_limit_seconds INTEGER",
            "ALTER TABLE exams ADD COLUMN allow_retries INTEGER NOT NULL DEFAULT 0",
        ],
    ),
    (
        "002_question_topics",
        [
            """CREATE TABLE IF NOT EXISTS topics (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL UNIQUE
            )""",
            """CREATE TABLE IF NOT EXISTS question_topics (
              question_id INTEGER NOT NULL,
              topic_id INTEGER NOT NULL,
              PRIMARY KEY (question_id, topic_id),
              FOREIGN KEY (question_id) REFERENCES exam_questions(id) ON DELETE CASCADE,
              FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
            )""",
        ],
    ),
    (
        "003_hints_used",
        [
            """CREATE TABLE IF NOT EXISTS hints_used (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              submission_id INTEGER NOT NULL,
              question_id INTEGER NOT NULL,
              line_index INTEGER NOT NULL,
              hint_text TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY (submission_id) REFERENCES submissions(id) ON DELETE CASCADE,
              FOREIGN KEY (question_id) REFERENCES exam_questions(id) ON DELETE CASCADE
            )"""
        ],
    ),
    (
        "004_grade_overrides",
        [
            "ALTER TABLE submission_lines ADD COLUMN override_correct INTEGER",
            "ALTER TABLE submission_lines ADD COLUMN override_reason TEXT",
            "ALTER TABLE submission_lines ADD COLUMN override_by INTEGER REFERENCES users(id)",
            "ALTER TABLE submission_lines ADD COLUMN override_at TEXT",
        ],
    ),
    (
        "005_partial_credit",
        [
            "ALTER TABLE submission_lines ADD COLUMN partial_score REAL NOT NULL DEFAULT 0",
        ],
    ),
    (
        "006_line_timing",
        [
            "ALTER TABLE submission_lines ADD COLUMN time_spent_ms INTEGER",
        ],
    ),
    (
        "007_ocr_confidence",
        [
            "ALTER TABLE submission_lines ADD COLUMN source TEXT NOT NULL DEFAULT 'typed'",
            "ALTER TABLE submission_lines ADD COLUMN ocr_confidence REAL",
        ],
    ),
    (
        "008_scratchpad",
        [
            """CREATE TABLE IF NOT EXISTS scratchpad (
              submission_id INTEGER NOT NULL,
              question_id INTEGER NOT NULL,
              content TEXT NOT NULL,
              updated_at TEXT NOT NULL DEFAULT (datetime('now')),
              PRIMARY KEY (submission_id, question_id),
              FOREIGN KEY (submission_id) REFERENCES submissions(id) ON DELETE CASCADE
            )"""
        ],
    ),
    (
        "009_spaced_repetition",
        [
            """CREATE TABLE IF NOT EXISTS srs_queue (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              question_id INTEGER NOT NULL,
              due_at TEXT NOT NULL,
              interval_days INTEGER NOT NULL DEFAULT 1,
              ease REAL NOT NULL DEFAULT 2.5,
              last_result TEXT,
              UNIQUE(user_id, question_id),
              FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
              FOREIGN KEY (question_id) REFERENCES exam_questions(id) ON DELETE CASCADE
            )"""
        ],
    ),
    (
        "010_exam_templates",
        [
            "ALTER TABLE exams ADD COLUMN cloned_from_id INTEGER REFERENCES exams(id)",
            "ALTER TABLE exams ADD COLUMN is_template INTEGER NOT NULL DEFAULT 0",
        ],
    ),
    (
        "011_rubrics",
        [
            """CREATE TABLE IF NOT EXISTS question_rubrics (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              question_id INTEGER NOT NULL,
              criterion TEXT NOT NULL,
              weight REAL NOT NULL DEFAULT 1.0,
              FOREIGN KEY (question_id) REFERENCES exam_questions(id) ON DELETE CASCADE
            )"""
        ],
    ),
]


def run_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_MIGRATIONS_TABLE)
    applied = {row["name"] for row in conn.execute("SELECT name FROM schema_migrations")}
    for name, statements in MIGRATIONS:
        if name in applied:
            continue
        for stmt in statements:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
        conn.execute("INSERT INTO schema_migrations (name) VALUES (?)", (name,))
