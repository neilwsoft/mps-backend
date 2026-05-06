"""FastAPI entrypoint for the Math Practice System.

Run locally with:

    fastapi dev main.py
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

import auth
import database
import gemini as gemini_client
from verifier import _normalize, verify_line


_ALLOWED_IMAGE_MIME = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
}
_MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    yield


app = FastAPI(title="Math Practice System", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- schemas --------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=6, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    name: str
    role: str


class AuthResponse(BaseModel):
    token: str
    user: UserOut


class QuestionPayload(BaseModel):
    prompt_latex: str = Field(min_length=1)
    solution_latex: list[str] = Field(min_length=1)


class CreateExamRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    questions: list[QuestionPayload] = Field(min_length=1)


class StartSubmissionResponse(BaseModel):
    submission_id: int
    exam_id: int


class SubmitLineRequest(BaseModel):
    question_id: int
    line_index: int = Field(ge=0)
    submitted_latex: str


class SubmitLineResponse(BaseModel):
    correct: bool
    explanation: str | None = None
    is_final_for_question: bool
    expected_total_lines: int


class FinalizeResponse(BaseModel):
    score: int
    total: int


class ExtractHandwritingResponse(BaseModel):
    lines: list[str]


# --- auth -----------------------------------------------------------------


def _user_out(user: dict) -> UserOut:
    return UserOut(
        id=user["id"], email=user["email"], name=user["name"], role=user["role"]
    )


@app.post("/api/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest) -> AuthResponse:
    existing = database.get_user_by_email(req.email)
    if existing is not None:
        raise HTTPException(status_code=409, detail="email already registered")
    uid = database.create_student(
        email=req.email,
        name=req.name.strip(),
        hashed_password=auth.hash_password(req.password),
    )
    user = database.get_user(uid)
    assert user is not None
    return AuthResponse(token=auth.encode_token(uid, "student"), user=_user_out(user))


@app.post("/api/auth/login", response_model=AuthResponse)
def login(req: LoginRequest) -> AuthResponse:
    record = database.get_user_by_email(req.email)
    if record is None or not auth.verify_password(req.password, record["hashed_password"]):
        raise HTTPException(status_code=401, detail="invalid email or password")
    user = database.get_user(record["id"])
    assert user is not None
    return AuthResponse(
        token=auth.encode_token(record["id"], record["role"]),
        user=_user_out(user),
    )


@app.get("/api/auth/me", response_model=UserOut)
def me(user: dict = Depends(auth.current_user)) -> UserOut:
    return _user_out(user)


# --- exams (read) ---------------------------------------------------------


@app.get("/api/exams")
def list_exams(user: dict = Depends(auth.current_user)) -> dict:
    return {"exams": database.list_exams()}


@app.get("/api/exams/{exam_id}")
def get_exam(exam_id: int, user: dict = Depends(auth.current_user)) -> dict:
    exam = database.get_exam(exam_id)
    if exam is None:
        raise HTTPException(status_code=404, detail="exam not found")
    # Students don't see model solutions while taking the exam.
    if user["role"] != "admin":
        exam = {
            **exam,
            "questions": [
                {
                    "id": q["id"],
                    "position": q["position"],
                    "prompt_latex": q["prompt_latex"],
                    "expected_line_count": len(q["solution_latex"]),
                }
                for q in exam["questions"]
            ],
        }
    return exam


# --- exams (admin write) --------------------------------------------------


@app.post("/api/exams")
def create_exam(
    req: CreateExamRequest, admin_user: dict = Depends(auth.require_admin)
) -> dict:
    exam_id = database.create_exam(
        title=req.title.strip(),
        description=req.description.strip(),
        created_by=admin_user["id"],
        questions=[q.model_dump() for q in req.questions],
    )
    return {"id": exam_id}


# --- submissions ----------------------------------------------------------


@app.post("/api/exams/{exam_id}/start", response_model=StartSubmissionResponse)
def start_submission(
    exam_id: int, user: dict = Depends(auth.current_user)
) -> StartSubmissionResponse:
    exam = database.get_exam(exam_id)
    if exam is None:
        raise HTTPException(status_code=404, detail="exam not found")
    sid = database.create_submission(exam_id=exam_id, user_id=user["id"])
    return StartSubmissionResponse(submission_id=sid, exam_id=exam_id)


def _ensure_owner_or_admin(submission: dict, user: dict) -> None:
    if user["role"] != "admin" and submission["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="not your submission")


@app.post(
    "/api/submissions/{submission_id}/lines",
    response_model=SubmitLineResponse,
)
def submit_line(
    submission_id: int,
    req: SubmitLineRequest,
    user: dict = Depends(auth.current_user),
) -> SubmitLineResponse:
    sub = database.get_submission(submission_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="submission not found")
    _ensure_owner_or_admin(sub, user)
    if sub["submitted_at"] is not None:
        raise HTTPException(status_code=409, detail="submission already finalized")

    question = database.get_exam_question(req.question_id)
    if question is None or question["exam_id"] != sub["exam_id"]:
        raise HTTPException(status_code=400, detail="question not in this exam")

    total = len(question["solution_latex"])
    if req.line_index >= total:
        raise HTTPException(status_code=400, detail="line_index out of range")

    expected = question["solution_latex"][req.line_index]
    result = verify_line(
        submitted_latex=req.submitted_latex,
        expected_latex=expected,
        problem_latex=question["prompt_latex"],
        line_index=req.line_index,
        total_lines=total,
    )

    database.record_submission_line(
        submission_id=submission_id,
        question_id=question["id"],
        line_index=req.line_index,
        submitted_latex=req.submitted_latex,
        correct=result.correct,
        explanation=result.explanation,
    )

    return SubmitLineResponse(
        correct=result.correct,
        explanation=result.explanation,
        is_final_for_question=req.line_index == total - 1,
        expected_total_lines=total,
    )


@app.post(
    "/api/submissions/{submission_id}/finalize",
    response_model=FinalizeResponse,
)
def finalize(
    submission_id: int, user: dict = Depends(auth.current_user)
) -> FinalizeResponse:
    sub = database.get_submission(submission_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="submission not found")
    _ensure_owner_or_admin(sub, user)
    if sub["submitted_at"] is not None:
        return FinalizeResponse(score=sub["score"], total=sub["total"])
    res = database.finalize_submission(submission_id)
    return FinalizeResponse(**res)


@app.get("/api/submissions")
def list_submissions(user: dict = Depends(auth.current_user)) -> dict:
    if user["role"] == "admin":
        return {"submissions": database.list_all_submissions()}
    return {"submissions": database.list_submissions_for_user(user["id"])}


@app.get("/api/submissions/{submission_id}")
def get_submission(
    submission_id: int, user: dict = Depends(auth.current_user)
) -> dict:
    sub = database.get_submission(submission_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="submission not found")
    _ensure_owner_or_admin(sub, user)
    return sub


# --- handwriting ---------------------------------------------------------


@app.post("/api/extract-handwriting", response_model=ExtractHandwritingResponse)
async def extract_handwriting(
    question_id: int = Form(...),
    image: UploadFile = File(...),
    user: dict = Depends(auth.current_user),
) -> ExtractHandwritingResponse:
    question = database.get_exam_question(question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="question not found")

    mime = (image.content_type or "").lower()
    if mime not in _ALLOWED_IMAGE_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported image type: {mime or 'unknown'}",
        )

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image upload")
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image too large (max 8MB)")

    lines = gemini_client.extract_handwritten_math(
        image_bytes=data,
        mime_type=mime,
        problem_latex=question["prompt_latex"],
        solution_latex=question["solution_latex"],
    )
    if lines and _normalize(lines[0]) == _normalize(question["prompt_latex"]):
        lines = lines[1:]

    if not lines:
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not read any math lines from the image. "
                "Make sure GEMINI_API_KEY is set and the photo is well-lit."
            ),
        )
    return ExtractHandwritingResponse(lines=lines)


# --- admin: students -----------------------------------------------------


@app.get("/api/admin/students")
def admin_list_students(_: dict = Depends(auth.require_admin)) -> dict:
    return {"students": database.list_students()}


@app.get("/api/admin/students/{user_id}")
def admin_student_detail(
    user_id: int, _: dict = Depends(auth.require_admin)
) -> dict:
    user = database.get_user(user_id)
    if user is None or user["role"] != "student":
        raise HTTPException(status_code=404, detail="student not found")
    return {
        "student": user,
        "submissions": database.list_submissions_for_user(user_id),
    }


@app.get("/api/admin/exams/{exam_id}/submissions")
def admin_exam_submissions(
    exam_id: int, _: dict = Depends(auth.require_admin)
) -> dict:
    exam = database.get_exam(exam_id)
    if exam is None:
        raise HTTPException(status_code=404, detail="exam not found")
    return {
        "exam": {"id": exam["id"], "title": exam["title"]},
        "submissions": database.list_submissions_for_exam(exam_id),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
