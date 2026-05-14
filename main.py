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
    time_spent_ms: int | None = Field(default=None, ge=0)
    source: str = Field(default="typed", pattern="^(typed|handwriting)$")
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    locale: str = Field(default="en", pattern="^(en|ko)$")


class SubmitLineResponse(BaseModel):
    correct: bool
    explanation: str | None = None
    is_final_for_question: bool
    expected_total_lines: int
    partial_score: float
    # The DB row id for this attempt. Used by the frontend to fetch a richer
    # Gemini-generated explanation via /api/submission-lines/{id}/explain
    # after the verdict has already been shown to the student.
    line_id: int


class ExplainLineRequest(BaseModel):
    locale: str = Field(default="en", pattern="^(en|ko)$")


class ExplainLineResponse(BaseModel):
    explanation: str


class HintRequest(BaseModel):
    question_id: int
    line_index: int = Field(ge=0)
    locale: str = Field(default="en", pattern="^(en|ko)$")


class HintResponse(BaseModel):
    hint: str


class ScratchpadRequest(BaseModel):
    content: str = Field(default="", max_length=20000)


class ScratchpadResponse(BaseModel):
    content: str


class OverrideLineRequest(BaseModel):
    correct: bool
    reason: str = Field(min_length=1, max_length=500)


class TopicTagsRequest(BaseModel):
    topics: list[str] = Field(default_factory=list)


class GenerateVariantsRequest(BaseModel):
    seed: QuestionPayload
    n: int = Field(default=3, ge=1, le=10)
    difficulty_delta: int = Field(default=0, ge=-1, le=1)


class GeneratedVariant(BaseModel):
    prompt_latex: str
    solution_latex: list[str]


class GenerateVariantsResponse(BaseModel):
    variants: list[GeneratedVariant]


class FinalizeResponse(BaseModel):
    score: int
    total: int


class ExtractedLine(BaseModel):
    latex: str
    confidence: float


class ExtractHandwritingResponse(BaseModel):
    lines: list[ExtractedLine]


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
        locale=req.locale,
    )

    line_id = database.record_submission_line(
        submission_id=submission_id,
        question_id=question["id"],
        line_index=req.line_index,
        submitted_latex=req.submitted_latex,
        correct=result.correct,
        explanation=result.explanation,
        partial_score=result.partial_score,
        time_spent_ms=req.time_spent_ms,
        source=req.source,
        ocr_confidence=req.ocr_confidence,
    )

    return SubmitLineResponse(
        correct=result.correct,
        explanation=result.explanation,
        is_final_for_question=req.line_index == total - 1,
        expected_total_lines=total,
        partial_score=result.partial_score,
        line_id=line_id,
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

    if not gemini_client.gemini_is_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Handwriting recognition isn't configured on the server. "
                "Please contact an administrator."
            ),
        )

    lines = gemini_client.extract_handwritten_math(
        image_bytes=data,
        mime_type=mime,
        problem_latex=question["prompt_latex"],
        solution_latex=question["solution_latex"],
    )
    # Drop a leading line that just restates the problem — but only when
    # there's another line after it. If the student wrote nothing but the
    # problem, keep that line so the verifier can tell them "this is still
    # the problem, do the next step."
    if (
        len(lines) > 1
        and _normalize(lines[0]["latex"]) == _normalize(question["prompt_latex"])
    ):
        lines = lines[1:]

    # No lines is a legitimate result (illegible handwriting, etc.), not a
    # server error — return an empty list and let the frontend show a clean
    # localized message that doesn't blame "the photo".
    return ExtractHandwritingResponse(
        lines=[ExtractedLine(latex=line["latex"], confidence=line["confidence"]) for line in lines]
    )


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


@app.get("/api/admin/students/{user_id}/analytics")
def admin_student_analytics(
    user_id: int, _: dict = Depends(auth.require_admin)
) -> dict:
    data = database.student_analytics(user_id)
    if not data:
        raise HTTPException(status_code=404, detail="student not found")
    return data


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


# --- async explanation ---------------------------------------------------


@app.post(
    "/api/submission-lines/{line_id}/explain",
    response_model=ExplainLineResponse,
)
def explain_line(
    line_id: int,
    req: ExplainLineRequest,
    user: dict = Depends(auth.current_user),
) -> ExplainLineResponse:
    """Fetch a richer, tutor-style explanation for a previously-recorded
    wrong submission line. The fast-path verifier doesn't call Gemini so the
    student gets a verdict instantly; this endpoint adds the personalized
    explanation in the background."""
    line = database.get_submission_line(line_id)
    if line is None:
        raise HTTPException(status_code=404, detail="line not found")
    if user["role"] != "admin" and line["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="not your submission line")
    if line["correct"]:
        # Nothing to explain — the student got it right.
        return ExplainLineResponse(explanation=line["explanation"] or "")

    expected = line["solution_latex"][line["line_index"]]
    rich = gemini_client.explain_mistake_detailed(
        problem=line["prompt_latex"],
        expected=expected,
        submitted=line["submitted_latex"],
        locale=req.locale,
    )
    explanation = rich or (line["explanation"] or "")
    if rich:
        database.update_line_explanation(line_id, rich)
    return ExplainLineResponse(explanation=explanation)


# --- hints ----------------------------------------------------------------


@app.post("/api/submissions/{submission_id}/hint", response_model=HintResponse)
def request_hint(
    submission_id: int,
    req: HintRequest,
    user: dict = Depends(auth.current_user),
) -> HintResponse:
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
    prior = database.get_prior_lines(submission_id, question["id"], req.line_index)

    if not gemini_client.gemini_is_configured():
        msg = (
            "힌트 기능을 사용하려면 Gemini API 키가 필요합니다. 백엔드에 "
            "GEMINI_API_KEY를 설정한 뒤 다시 시도해 주세요."
            if req.locale == "ko"
            else (
                "Hints require a Gemini API key. Set GEMINI_API_KEY on the "
                "backend, then try again."
            )
        )
        raise HTTPException(status_code=503, detail=msg)

    hint = gemini_client.generate_hint(
        problem=question["prompt_latex"],
        expected=expected,
        prior_lines=prior,
        step_num=req.line_index + 1,
        total_steps=total,
        locale=req.locale,
    )

    # If Gemini is configured but failed (rate-limit, parse error, or returned a
    # generic answer), fall back to a sympy-driven hint that references the actual
    # expected step and last student line. This stays specific to the problem.
    if not hint or _is_generic(hint, req.locale):
        hint = _heuristic_hint(
            problem=question["prompt_latex"],
            expected=expected,
            last_line=prior[-1] if prior else question["prompt_latex"],
            locale=req.locale,
        )

    database.record_hint(submission_id, question["id"], req.line_index, hint)
    return HintResponse(hint=hint)


_GENERIC_HINT_MARKERS = {
    "en": ("look at the operation", "what should you do", "move forward"),
    "ko": ("연산을 보세요", "무엇을 해야 할까요", "앞으로 나아가"),
}


def _is_generic(text: str, locale: str = "en") -> bool:
    low = text.lower()
    return any(m in low for m in _GENERIC_HINT_MARKERS.get(locale, _GENERIC_HINT_MARKERS["en"]))


_HEURISTIC_HINT_MESSAGES = {
    "en": {
        "matches_value": "Your last line already matches the value of the next step — rewrite it in the form the worksheet expects.",
        "the_variable": "the variable {sym}",
        "the_expression": "the expression",
        "combine_like": "Combine the like terms involving {sym} from your last line so the highest power drops by one.",
        "multiply_out": "Multiply out the brackets in your last line so {sym} appears at a higher power.",
        "operation": "Look at how {sym} changes between your last line and the target — apply that operation to every term.",
        "reread": "Re-read your last step out loud and check whether each term has been simplified — a coefficient or sign may need to move.",
    },
    "ko": {
        "matches_value": "마지막 줄의 값은 이미 다음 단계와 같습니다 — 학습지가 요구하는 형태로 다시 적어 보세요.",
        "the_variable": "변수 {sym}",
        "the_expression": "이 식",
        "combine_like": "마지막 줄에서 {sym}이(가) 들어간 동류항을 합쳐 최고차항의 차수를 한 단계 낮춰 보세요.",
        "multiply_out": "마지막 줄의 괄호를 풀어서 {sym}이(가) 더 높은 차수로 나오게 만들어 보세요.",
        "operation": "마지막 줄에서 목표 식까지 {sym}이(가) 어떻게 바뀌는지 보고, 그 연산을 모든 항에 적용해 보세요.",
        "reread": "마지막 단계를 소리 내어 다시 읽고 각 항이 충분히 정리되었는지 확인하세요 — 계수나 부호가 옮겨져야 할 수 있습니다.",
    },
}


def _heuristic_hint(
    problem: str, expected: str, last_line: str, locale: str = "en"
) -> str:
    """Deterministic, specific-to-the-problem hint when Gemini is unavailable
    or returned a generic answer. We use sympy to compare the previous student
    line to the expected step and infer the operation.
    """
    msgs = _HEURISTIC_HINT_MESSAGES.get(locale, _HEURISTIC_HINT_MESSAGES["en"])
    try:
        import sympy
        from verifier import _parse  # type: ignore

        prev_expr = _parse(last_line)
        target_expr = _parse(expected)
        diff = sympy.simplify(target_expr - prev_expr)
        if diff == 0:
            return msgs["matches_value"]
        free_syms = sorted({str(s) for s in target_expr.free_symbols})
        sym_hint = (
            msgs["the_variable"].format(sym=free_syms[0])
            if free_syms
            else msgs["the_expression"]
        )
        try:
            prev_poly = sympy.Poly(prev_expr)
            target_poly = sympy.Poly(target_expr)
            if prev_poly.degree() > target_poly.degree():
                return msgs["combine_like"].format(sym=sym_hint)
            if prev_poly.degree() < target_poly.degree():
                return msgs["multiply_out"].format(sym=sym_hint)
        except Exception:
            pass
        return msgs["operation"].format(sym=sym_hint)
    except Exception:
        return msgs["reread"]


# --- scratchpad -----------------------------------------------------------


@app.get(
    "/api/submissions/{submission_id}/scratchpad/{question_id}",
    response_model=ScratchpadResponse,
)
def get_scratchpad(
    submission_id: int,
    question_id: int,
    user: dict = Depends(auth.current_user),
) -> ScratchpadResponse:
    sub = database.get_submission(submission_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="submission not found")
    _ensure_owner_or_admin(sub, user)
    return ScratchpadResponse(content=database.get_scratchpad(submission_id, question_id))


@app.put(
    "/api/submissions/{submission_id}/scratchpad/{question_id}",
    response_model=ScratchpadResponse,
)
def put_scratchpad(
    submission_id: int,
    question_id: int,
    req: ScratchpadRequest,
    user: dict = Depends(auth.current_user),
) -> ScratchpadResponse:
    sub = database.get_submission(submission_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="submission not found")
    _ensure_owner_or_admin(sub, user)
    database.upsert_scratchpad(submission_id, question_id, req.content)
    return ScratchpadResponse(content=req.content)


# --- admin: grade override ------------------------------------------------


@app.patch("/api/admin/submission-lines/{line_id}")
def admin_override_line(
    line_id: int,
    req: OverrideLineRequest,
    admin_user: dict = Depends(auth.require_admin),
) -> dict:
    result = database.override_submission_line(
        line_id=line_id,
        override_correct=req.correct,
        reason=req.reason.strip(),
        admin_user_id=admin_user["id"],
    )
    if result is None:
        raise HTTPException(status_code=404, detail="submission line not found")
    return {"score": result["score"], "total": result["total"]}


# --- admin: AI question generation ---------------------------------------


@app.post(
    "/api/admin/questions/generate", response_model=GenerateVariantsResponse
)
def admin_generate_variants(
    req: GenerateVariantsRequest,
    _: dict = Depends(auth.require_admin),
) -> GenerateVariantsResponse:
    variants = gemini_client.generate_question_variants(
        seed_prompt=req.seed.prompt_latex,
        seed_solution=req.seed.solution_latex,
        n=req.n,
        difficulty_delta=req.difficulty_delta,
    )
    if not variants:
        raise HTTPException(
            status_code=502,
            detail="Could not generate variants — make sure GEMINI_API_KEY is set.",
        )
    return GenerateVariantsResponse(
        variants=[
            GeneratedVariant(prompt_latex=v["prompt_latex"], solution_latex=v["solution_latex"])
            for v in variants
        ]
    )


# --- admin: exam clone ---------------------------------------------------


@app.post("/api/admin/exams/{exam_id}/clone")
def admin_clone_exam(
    exam_id: int,
    admin_user: dict = Depends(auth.require_admin),
) -> dict:
    new_id = database.clone_exam(exam_id, admin_user["id"])
    if new_id is None:
        raise HTTPException(status_code=404, detail="exam not found")
    return {"id": new_id}


# --- topics & analytics --------------------------------------------------


@app.get("/api/admin/topics")
def admin_list_topics(_: dict = Depends(auth.require_admin)) -> dict:
    return {"topics": database.list_topics()}


@app.put("/api/admin/questions/{question_id}/topics")
def admin_set_topics(
    question_id: int,
    req: TopicTagsRequest,
    _: dict = Depends(auth.require_admin),
) -> dict:
    if database.get_exam_question(question_id) is None:
        raise HTTPException(status_code=404, detail="question not found")
    database.set_question_topics(question_id, req.topics)
    return {"topics": database.get_question_topics(question_id)}


@app.get("/api/admin/exams/{exam_id}/metrics")
def admin_exam_metrics(
    exam_id: int, _: dict = Depends(auth.require_admin)
) -> dict:
    if database.get_exam(exam_id) is None:
        raise HTTPException(status_code=404, detail="exam not found")
    return {"metrics": database.question_metrics(exam_id)}


@app.get("/api/admin/analytics/topic-mastery")
def admin_topic_mastery(_: dict = Depends(auth.require_admin)) -> dict:
    return {"rows": database.topic_mastery()}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
