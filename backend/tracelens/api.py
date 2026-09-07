"""FastAPI application and HTTP route composition."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from starlette.concurrency import run_in_threadpool
from . import agent as agent_module
from .auth import bearer, create_access_token, current_user, enforce_login_rate_limit, hash_password, optional_user, require_authenticated_user, revoke_access_token, verify_password
from .bilibili import BilibiliImportError, download_bilibili_video, fetch_bilibili_metadata, normalize_bvid
from .agent import (
    AgentToolbox,
    answer_from_evidence,
    evidence_citations,
    is_summary_goal,
    llm_settings,
    parse_kimi_json,
    run_structured_deepseek_agent as _run_structured_deepseek_agent,
    run_deepseek_follow_up,
)
from .agent_graph import AGENT_TOOL_NAMES
from .config import env_flag, logger
from . import ocr_runner
from .ocr_runner import extract_ocr_evidence, extract_transcript_from_video, merge_transcript_chunks, resolve_ffmpeg_path
from .media import validate_video_filename, validate_video_id
from .models import AskIn, AuthIn, BilibiliIn, DemoSeedIn, EvidenceIn
from .retrieval import (
    _QDRANT_CLIENT,
    ensure_qdrant_collection,
    evidence_window,
    get_qdrant_client,
    get_embedding_model,
    parse_time_hints,
    search_evidence,
    search_timeline,
    sync_evidence_to_qdrant,
)
from .storage import (
    DB_PATH,
    UPLOADS_DIR,
    create_media_task,
    create_user,
    claim_video,
    get_connection,
    get_latest_agent_result,
    get_media_task,
    get_user_by_username,
    user_owns_video,
    get_or_create_session,
    get_session_history,
    get_session_summary,
    init_db,
    register_legacy_videos,
    upsert_video,
    save_agent_turn,
    update_media_task,
    update_media_task as persist_media_task,
)
from .tasks import submit_task

OpenAI = agent_module.OpenAI
_agent_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="report-agent")


def deepseek_is_configured() -> bool:
    agent_module.OpenAI = OpenAI
    return agent_module.deepseek_is_configured()


kimi_is_configured = deepseek_is_configured


def run_deepseek_agent(question: str, video_id: str | None, history: list[dict[str, str]] | None = None) -> dict[str, Any] | None:
    agent_module.OpenAI = OpenAI
    if os.getenv("DEEPSEEK_AGENT_WORKFLOW", os.getenv("KIMI_AGENT_WORKFLOW", "structured")).strip().lower() in {
        "structured",
        "planner-retriever-verifier-writer-critic",
    }:
        structured_result = _run_structured_deepseek_agent(question, video_id)
        if structured_result is not None:
            return structured_result
    return agent_module.run_kimi_agent(question, video_id, history=history)


def local_report_fallback(question: str, video_id: str | None, error: str) -> dict[str, Any]:
    all_evidence = agent_module.load_video_evidence(video_id)
    asr_evidence = [item for item in all_evidence if item.source.upper() == "ASR" and len(item.text.strip()) >= 8]
    ocr_evidence = [item for item in all_evidence if item.source.upper() == "OCR" and len(item.text.strip()) >= 12]
    # Keep the offline path consistent with the structured report: OCR is
    # preferred for visible claims, while ASR remains available as context.
    ocr_selected = ocr_evidence[: min(len(ocr_evidence), 6)]
    asr_selected = asr_evidence[: max(0, 9 - len(ocr_selected))]
    evidence = sorted(
        ocr_selected + asr_selected,
        key=lambda item: (item.start_seconds, item.id),
    )[:9]
    if not evidence:
        return {
            "question": question,
            "answer": "报告生成失败，当前视频没有可用证据。",
            "grounded": False,
            "citations": [],
            "provider": "DeepSeek error",
            "error": str(error)[:500],
            "trace": ["DeepSeek structured report failed", "No persisted evidence for fallback"],
        }
    citations = evidence_citations(evidence)
    answer = (
        "DeepSeek 暂时不可用，以下为已保存视频证据的本地摘要："
        + "；".join(item.text for item in evidence)
        if evidence
        else "当前视频没有已保存的可用证据。"
    )
    return {
        "question": question,
        "answer": answer,
        "grounded": bool(evidence),
        "citations": citations,
        "provider": "local evidence fallback",
        "kind": "initial_report",
        "support_level": "SUMMARY" if evidence else "INSUFFICIENT",
        "error": str(error)[:500],
        "report": {
            "answerable": bool(evidence),
            "title": "本地证据摘要",
            "finalAnswer": answer,
            "conclusions": [item.text for item in evidence[:5]],
            "evidence": citations,
            "suggestions": ["稍后可重新提交问题以尝试生成 DeepSeek 结构化报告。"],
        },
        "trace": ["DeepSeek structured report failed", "Fallback: summarized persisted ASR/OCR evidence"],
        "tool_trace": [],
    }


def local_follow_up_fallback(question: str, video_id: str | None) -> dict[str, Any]:
    selected = AgentToolbox(video_id, question).evidence_for_question(question, max_segments=8)
    answer = (
        "根据已保存的视频证据：" + "；".join(
            str(item.get("content", item.get("text", ""))).strip()
            for item in selected[:4]
        )
        if selected else "当前视频证据不足，暂时无法确定这个追问的答案。"
    )
    return {
        "question": question,
        "answer": answer,
        "grounded": bool(selected),
        "citations": [],
        "provider": "local follow-up fallback",
        "kind": "follow_up",
        "support_level": "DIRECT" if selected else "INSUFFICIENT",
        "trace": ["DeepSeek follow-up unavailable", "Fallback: selected persisted evidence"],
        "tool_trace": [],
    }


def run_agent_with_timeout(
    question: str,
    video_id: str | None,
    history: list[dict[str, str]],
    is_follow_up: bool,
    session_id: str,
) -> dict[str, Any] | None:
    # The structured workflow makes several bounded model calls. Allow a
    # cold provider connection to complete instead of masking it as fallback.
    timeout_seconds = max(20, float(os.getenv("REPORT_TIMEOUT_SECONDS", "180")))
    if is_follow_up:
        previous_report = get_latest_agent_result(session_id) or {}
        future = _agent_executor.submit(
            lambda: run_deepseek_follow_up(
                question,
                video_id,
                previous_report or {},
                history=history,
                memory_summary=get_session_summary(session_id),
            )
        )
    else:
        future = _agent_executor.submit(run_deepseek_agent, question, video_id, history)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        logger.warning("report_agent_timeout follow_up=%s timeout_seconds=%s", is_follow_up, timeout_seconds)
        return {
            "question": question,
            "answer": "DeepSeek report generation timed out. Please retry after checking the backend log.",
            "grounded": False,
            "provider": "DeepSeek timeout",
            "error": "report generation timed out",
            "kind": "follow_up" if is_follow_up else "initial_report",
            "trace": ["DeepSeek report exceeded the server time budget"],
        }


run_kimi_agent = run_deepseek_agent


def _run_analysis_task(task_id: str, question: str, video_id: str | None, session_id: str) -> dict[str, Any]:
    """Run one initial report outside the HTTP request lifecycle."""
    update_media_task(
        task_id,
        progress_current=1,
        progress_total=5,
        progress_message="正在理解问题并规划证据需求",
    )
    update_media_task(
        task_id,
        progress_current=2,
        progress_total=5,
        progress_message="正在从 ASR/OCR 检索时间证据",
    )
    result = run_deepseek_agent(question, video_id, get_session_history(session_id))
    if not result:
        result = {
            "question": question,
            "answer": "当前未获得模型返回，无法从视频确定答案。",
            "grounded": False,
            "provider": "DeepSeek unavailable",
            "error": "DeepSeek 未返回报告",
        }
    if result.get("error") and not result.get("report"):
        answer = str(result.get("answer") or "模型暂时不可用，未能完成本次视频分析。")
        result["report"] = {
            "answerable": False,
            "title": "视频分析未完成",
            "finalAnswer": answer,
            "conclusions": [answer],
            "evidence": [],
            "suggestions": ["请检查模型配置或稍后重新生成报告。"],
        }
        result["kind"] = "initial_report"
    result["session_id"] = session_id
    update_media_task(
        task_id,
        progress_current=4,
        progress_total=5,
        progress_message="正在校验引用并整理最终报告",
    )
    save_agent_turn(session_id, question, result)
    update_media_task(
        task_id,
        progress_current=5,
        progress_total=5,
        progress_message="报告生成完成",
    )
    return result


def _analysis_task_response(task: dict[str, Any]) -> dict[str, Any]:
    result = None
    if task.get("result_json"):
        try:
            parsed = json.loads(task["result_json"])
            result = parsed if isinstance(parsed, dict) else None
        except (TypeError, json.JSONDecodeError):
            result = None
    total = max(0, int(task.get("progress_total") or 0))
    current = max(0, int(task.get("progress_current") or 0))
    return {
        "taskId": task["task_id"],
        "videoId": task.get("video_id"),
        "sessionId": task.get("session_id"),
        "question": task.get("question"),
        "taskType": task.get("task_type"),
        "state": task.get("state"),
        "progressCurrent": current,
        "progressTotal": total,
        "progressPercent": min(100, round(current * 100 / total)) if total else 0,
        "stage": task.get("progress_message") or task.get("state"),
        "message": task.get("progress_message") or task.get("error") or task.get("state"),
        "error": task.get("error"),
        "result": result,
        "report": result,
    }


def _quick_analysis_result(task_id: str, seconds: float = 0.15) -> dict[str, Any] | None:
    """Keep fast local/test providers compatible while real calls stay async."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        task = get_media_task(task_id)
        if task and task["state"] in {"COMPLETED", "FAILED"}:
            return _analysis_task_response(task)
        time.sleep(0.01)
    return None


def _start_analysis_task(question: str, video_id: str | None, session_id: str) -> dict[str, Any]:
    """Create/deduplicate an initial-report task and dispatch the local worker."""
    with get_connection() as connection:
        active = connection.execute(
            "SELECT * FROM media_tasks WHERE task_type = 'ANALYSIS' "
            "AND (video_id = ? OR (video_id IS NULL AND ? IS NULL)) "
            "AND question = ? AND state IN ('QUEUED', 'RUNNING') "
            "ORDER BY created_at DESC LIMIT 1",
            (video_id, video_id, question),
        ).fetchone()
    if active:
        return _analysis_task_response(dict(active))
    task_id = create_media_task(
        video_id or "",
        "ANALYSIS",
        question=question,
        session_id=session_id,
    )
    submit_task(
        task_id,
        lambda: _run_analysis_task(task_id, question, video_id, session_id),
    )
    immediate = _quick_analysis_result(task_id)
    return immediate or _analysis_task_response(get_media_task(task_id) or {
        "task_id": task_id,
        "video_id": video_id,
        "session_id": session_id,
        "question": question,
        "task_type": "ANALYSIS",
        "state": "QUEUED",
        "progress_message": "任务已提交，等待分析",
    })


def _validate_analysis_video(video_id: str | None) -> None:
    if not video_id:
        return
    with get_connection() as connection:
        video = connection.execute(
            "SELECT status FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
    if video is not None and str(video["status"]).upper() not in {"COMPLETED"}:
        raise HTTPException(
            status_code=409,
            detail=f"video is not ready for analysis: {video['status']}",
        )


def _create_initial_analysis(payload: AskIn) -> dict[str, Any]:
    _validate_analysis_video(payload.video_id)
    session_id = get_or_create_session(payload.session_id, payload.video_id)
    response = {
        **_start_analysis_task(payload.question.strip(), payload.video_id, session_id),
        "session_id": session_id,
        "accepted": True,
    }
    # Preserve the old API shape when a fast local/test provider has already
    # completed during the short hand-off window. Real provider calls return
    # the task envelope immediately and are polled by the client.
    if response.get("state") == "COMPLETED" and isinstance(response.get("result"), dict):
        return {**response["result"], **response}
    return response


def _resume_pending_analysis_tasks() -> None:
    """Re-queue persisted local tasks after a backend restart."""
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM media_tasks WHERE task_type = 'ANALYSIS' AND state IN ('QUEUED', 'RUNNING')"
        ).fetchall()
    for row in rows:
        task = dict(row)
        if not task.get("question") or not task.get("session_id"):
            update_media_task(task["task_id"], state="FAILED", error="分析任务缺少问题或会话信息")
            continue
        update_media_task(
            task["task_id"],
            state="QUEUED",
            progress_message="后端已重启，正在恢复分析任务",
        )
        submit_task(
            task["task_id"],
            lambda task=task: _run_analysis_task(
                task["task_id"], task["question"], task.get("video_id") or None, task["session_id"]
            ),
        )


def _is_invalid_persisted_report(report_json: str | None) -> bool:
    """Reject only malformed legacy rows; failures remain visible records."""
    raw = parse_kimi_json(report_json or "") or {}
    return not raw
app = FastAPI(title="TraceLens", version="0.4.0")
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup() -> None:
    init_db()
    register_legacy_videos()
    _resume_pending_analysis_tasks()


@app.on_event("shutdown")
def shutdown() -> None:
    if _QDRANT_CLIENT is not None:
        try:
            _QDRANT_CLIENT.close()
        except Exception:
            logger.exception("Unable to close Qdrant")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "tracelens"}


@app.post("/api/auth/register")
def register(payload: AuthIn, request: Request) -> dict[str, Any]:
    enforce_login_rate_limit(request, f"register:{payload.username}")
    try:
        user = create_user(payload.username, hash_password(payload.password))
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"user": user, "access_token": create_access_token(user), "token_type": "bearer"}


@app.post("/api/auth/login")
def login(payload: AuthIn, request: Request) -> dict[str, Any]:
    enforce_login_rate_limit(request, payload.username)
    user = get_user_by_username(payload.username)
    if user is None or not verify_password(payload.password, str(user.get("password_hash", ""))):
        raise HTTPException(status_code=401, detail="invalid username or password")
    return {"user": {"id": user["id"], "username": user["username"], "role": user["role"]},
            "access_token": create_access_token(user), "token_type": "bearer"}


@app.post("/api/auth/logout")
def logout(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict[str, bool]:
    if credentials is not None:
        try:
            revoke_access_token(credentials.credentials)
        except HTTPException:
            pass
    return {"logged_out": True}


@app.get("/api/auth/me")
def me(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    return {"id": user["id"], "username": user["username"], "role": user["role"]}


def _authorize_video(video_id: str | None, user: dict[str, Any] | None) -> None:
    if not env_flag("AUTH_REQUIRED", True):
        return
    if user is None:
        raise HTTPException(status_code=401, detail="authentication required")
    if not video_id:
        return
    if not user_owns_video(video_id, int(user["id"])):
        raise HTTPException(status_code=404, detail="video not found")


@app.post("/api/media/bilibili/preview")
def preview_bilibili(payload: BilibiliIn, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    require_authenticated_user(user)
    try:
        return fetch_bilibili_metadata(payload.bvid)
    except (ValueError, BilibiliImportError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _run_bilibili_import_task(task_id: str, bvid: str) -> dict[str, Any]:
    update_media_task(
        task_id,
        progress_current=1,
        progress_total=4,
        progress_message="Downloading Bilibili video",
    )
    video_dir = UPLOADS_DIR / bvid
    downloaded = download_bilibili_video(bvid, video_dir)
    max_duration = int(os.getenv("BILIBILI_MAX_DURATION_SECONDS", "900"))
    max_bytes = int(os.getenv("BILIBILI_MAX_FILE_BYTES", str(512 * 1024 * 1024)))
    if downloaded.duration_seconds > max_duration:
        downloaded.path.unlink(missing_ok=True)
        raise ValueError(f"视频时长超过限制（最多 {max_duration} 秒）")
    if downloaded.path.stat().st_size > max_bytes:
        downloaded.path.unlink(missing_ok=True)
        raise ValueError(f"视频文件超过限制（最多 {max_bytes} bytes）")
    with get_connection() as connection:
        upsert_video(
            connection,
            {"video_id": bvid, "filename": downloaded.path.name, "stored_path": str(downloaded.path),
             "content_hash": hashlib.sha256(downloaded.path.read_bytes()).hexdigest(),
             "status": "UPLOADED", "ocr_status": "QUEUED"},
            ["filename", "stored_path", "content_hash", "status", "ocr_status"],
        )
    result = _run_transcription_task(
        task_id,
        bvid,
        downloaded.path,
        downloaded.path.name,
        progress_offset=1,
        progress_total=4,
    )
    return {"bvid": bvid, "title": downloaded.title, "uploader": downloaded.uploader, **result}


@app.post("/api/media/bilibili/import")
def import_bilibili(payload: BilibiliIn, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    require_authenticated_user(user)
    if not env_flag("BILIBILI_IMPORT_ENABLED", True):
        raise HTTPException(status_code=503, detail="Bilibili import is disabled")
    try:
        bvid = normalize_bvid(payload.bvid)
        metadata = fetch_bilibili_metadata(bvid)
    except (ValueError, BilibiliImportError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    max_duration = int(os.getenv("BILIBILI_MAX_DURATION_SECONDS", "900"))
    if int(metadata.get("durationSeconds") or 0) > max_duration:
        raise HTTPException(status_code=422, detail=f"视频时长超过限制（最多 {max_duration} 秒）")
    if env_flag("AUTH_REQUIRED", True):
        if user is None:
            raise HTTPException(status_code=401, detail="authentication required")
        claim_video(bvid, int(user["id"]))
    task_id = create_media_task(bvid, "BILIBILI_IMPORT")
    submit_task(task_id, lambda: _run_bilibili_import_task(task_id, bvid))
    return {"bvid": bvid, "metadata": metadata, "status": "queued", "task_id": task_id}


@app.get("/api/media/bilibili/import-status")
def bilibili_import_status(task_id: str, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    task = get_media_task(task_id)
    if task is None or str(task.get("task_type", "")).upper() != "BILIBILI_IMPORT":
        raise HTTPException(status_code=404, detail="bilibili import task not found")
    _authorize_video(str(task.get("video_id") or ""), user)
    return get_task(task_id, user=user)


@app.get("/api/evidence")
def list_evidence(video_id: str | None = None, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    _authorize_video(video_id, user)
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, video_id, start_seconds, end_seconds, text, source FROM evidence "
            "WHERE (? IS NULL OR video_id = ?) ORDER BY start_seconds, id",
            (video_id, video_id),
        ).fetchall()
    return {"items": [dict(row) for row in rows]}


@app.get("/api/evidence/search")
def search_evidence_api(
    query: str,
    video_id: str | None = None,
    top_k: int = 8,
    sources: str | None = None,
    user: dict[str, Any] | None = Depends(optional_user),
) -> dict[str, Any]:
    _authorize_video(video_id, user)
    source_list = [item.strip().upper() for item in (sources or "").split(",") if item.strip()]
    return search_timeline(query, video_id, top_k, source_list)


@app.get("/api/evidence/window")
def evidence_window_api(
    video_id: str,
    timestamp_ms: int,
    before_ms: int = 15000,
    after_ms: int = 15000,
    user: dict[str, Any] | None = Depends(optional_user),
) -> dict[str, Any]:
    _authorize_video(video_id, user)
    return evidence_window(video_id, timestamp_ms, before_ms, after_ms)


@app.get("/api/evidence/time-hints")
def evidence_time_hints(query: str) -> dict[str, Any]:
    return {"query": query, "timestampsMs": parse_time_hints(query)}


@app.post("/api/evidence")
def create_evidence(payload: EvidenceIn, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    if env_flag("AUTH_REQUIRED", True):
        if user is None:
            raise HTTPException(status_code=401, detail="authentication required")
        claim_video(payload.video_id, int(user["id"]))
    if payload.end_seconds <= payload.start_seconds:
        raise HTTPException(status_code=422, detail="end_seconds must be greater than start_seconds")
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO evidence(video_id, start_seconds, end_seconds, text, source) VALUES (?, ?, ?, ?, 'ASR')",
            (payload.video_id, payload.start_seconds, payload.end_seconds, payload.text),
        )
        evidence_id = cursor.lastrowid
    sync_evidence_to_qdrant(payload.video_id)
    return {"id": evidence_id, **payload.model_dump()}


@app.post("/api/demo/seed")
def seed_demo(payload: DemoSeedIn, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    if env_flag("AUTH_REQUIRED", True):
        if user is None:
            raise HTTPException(status_code=401, detail="authentication required")
        claim_video(payload.video_id, int(user["id"]))
    demo_items = [
        (0, 42, "视频先介绍了项目式学习：先做一个能运行的小项目，再围绕项目补齐知识。"),
        (48, 96, "视频提到面试准备要围绕自己的项目，重点讲清楚数据流、技术选型和故障排查。"),
        (105, 158, "视频总结了证据链：视频转写得到带时间戳的文本，检索后再生成带引用的回答。"),
    ]
    with get_connection() as connection:
        connection.execute("DELETE FROM evidence WHERE video_id = ?", (payload.video_id,))
        upsert_video(
            connection,
            {"video_id": payload.video_id, "filename": f"{payload.video_id}.demo.mp4",
             "stored_path": "", "content_hash": f"demo:{payload.video_id}",
             "status": "COMPLETED", "ocr_status": "DISABLED"},
            ["status", "ocr_status"],
        )
        connection.executemany(
            "INSERT INTO evidence(video_id, start_seconds, end_seconds, text, source) VALUES (?, ?, ?, ?, 'ASR')",
            [(payload.video_id, start, end, text) for start, end, text in demo_items],
        )
    sync_evidence_to_qdrant(payload.video_id)
    return {"video_id": payload.video_id, "seeded": len(demo_items)}


@app.get("/api/videos")
def list_videos(user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    require_authenticated_user(user)
    owner_id = int(user["id"]) if env_flag("AUTH_REQUIRED", True) and user else None
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT v.video_id, v.filename, v.content_hash, v.status, v.ocr_status, "
            "v.transcript_text IS NOT NULL AS has_transcript, v.created_at, v.updated_at, "
            "COUNT(e.id) AS evidence_count "
            "FROM videos v LEFT JOIN evidence e ON e.video_id = v.video_id "
            "WHERE (? IS NULL OR EXISTS (SELECT 1 FROM video_owners o WHERE o.video_id = v.video_id AND o.user_id = ?)) "
            "GROUP BY v.video_id ORDER BY v.updated_at DESC, v.video_id"
            , (owner_id, owner_id)).fetchall()
    return {"items": [dict(row) for row in rows]}


def _run_transcription_task(
    task_id: str,
    video_id: str,
    stored_path: Path,
    filename: str,
    *,
    progress_offset: int = 0,
    progress_total: int = 3,
) -> dict[str, Any]:
    persist_progress = persist_media_task

    def update_media_task(task_id: str, **values: Any) -> None:
        values["progress_current"] = int(values.get("progress_current", 0)) + progress_offset
        values["progress_total"] = progress_total
        persist_progress(task_id, **values)

    update_media_task(task_id, progress_current=1, progress_total=3, progress_message="正在提取音频并转录")
    transcript: list[tuple[float, float, str]] = []
    asr_error = ""
    try:
        transcript = extract_transcript_from_video(stored_path, filename)
    except Exception as error:
        asr_error = str(error)
        logger.warning("ASR failed for task %s: %s", task_id, error)

    ocr_evidence: list[tuple[float, float, str]] = []
    ocr_error = ""
    if env_flag("OCR_ENABLED", True):
        update_media_task(task_id, progress_current=2, progress_total=3, progress_message="正在提取画面文字（OCR）")
        try:
            ocr_evidence = extract_ocr_evidence(stored_path, filename)
        except Exception as error:
            ocr_error = str(error)
            logger.warning("OCR failed for task %s: %s", task_id, error)

    if not transcript and not ocr_evidence:
        raise RuntimeError(
            "ASR and OCR produced no evidence"
            + (f"; ASR error: {asr_error[:300]}" if asr_error else "")
            + (f"; OCR error: {ocr_error[:300]}" if ocr_error else "")
        )

    with get_connection() as connection:
        connection.execute("DELETE FROM evidence WHERE video_id = ?", (video_id,))
        connection.executemany(
            "INSERT INTO evidence(video_id, start_seconds, end_seconds, text, source) VALUES (?, ?, ?, ?, ?)",
            [(video_id, start, end, text, "ASR") for start, end, text in transcript]
            + [(video_id, start, end, text, "OCR") for start, end, text in ocr_evidence],
        )
        connection.execute(
            "UPDATE videos SET status = 'COMPLETED', ocr_status = ?, transcript_text = ?, updated_at = CURRENT_TIMESTAMP WHERE video_id = ?",
            (
                "DISABLED" if not env_flag("OCR_ENABLED", True) else "COMPLETED" if ocr_evidence else "FAILED" if ocr_error else "COMPLETED",
                "\n".join(item[2] for item in transcript),
                video_id,
            ),
        )
    sync_evidence_to_qdrant(video_id)
    update_media_task(task_id, progress_current=3, progress_total=3, progress_message="转录与证据整理完成")
    return {
        "video_id": video_id,
        "evidence_count": len(transcript) + len(ocr_evidence),
        "asr_count": len(transcript),
        "ocr_count": len(ocr_evidence),
        **({"ocr_error": ocr_error} if ocr_error else {}),
    }


@app.post("/api/videos/upload")
async def upload_video(
    video_id: str = Form(..., min_length=1),
    file: UploadFile = File(...),
    background: bool = Form(False),
    user: dict[str, Any] | None = Depends(optional_user),
) -> dict[str, Any]:
    video_id = validate_video_id(video_id)
    if env_flag("AUTH_REQUIRED", True):
        if user is None:
            raise HTTPException(status_code=401, detail="authentication required")
        claim_video(video_id, int(user["id"]))
    if not file.filename:
        raise HTTPException(status_code=400, detail="file name is required")

    safe_name = validate_video_filename(file.filename)
    video_dir = UPLOADS_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    stored_path = video_dir / safe_name
    temporary_path = video_dir / f".{uuid.uuid4().hex}.uploading"
    max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
    total_bytes = 0
    digest = hashlib.sha256()
    try:
        with temporary_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_upload_bytes:
                    raise HTTPException(status_code=413, detail="video file is too large")
                digest.update(chunk)
                output.write(chunk)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    content_hash = digest.hexdigest()
    existing: Any | None = None
    with get_connection() as connection:
        existing = connection.execute(
            "SELECT video_id, filename, stored_path, content_hash, ocr_status FROM videos WHERE video_id = ?",
            (video_id,),
        ).fetchone()
        evidence_count = connection.execute(
            "SELECT COUNT(*) FROM evidence WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
        ocr_count = connection.execute(
            "SELECT COUNT(*) FROM evidence WHERE video_id = ? AND source = 'OCR'",
            (video_id,),
        ).fetchone()[0]
    ocr_enabled = env_flag("OCR_ENABLED", True)
    if (
        existing is not None
        and existing["content_hash"] == content_hash
        and evidence_count > 0
        and (
            not ocr_enabled
            or ocr_count > 0
            or str(existing["ocr_status"] or "UNKNOWN").upper() in {"COMPLETED", "FAILED"}
        )
        and Path(existing["stored_path"]).is_file()
    ):
        temporary_path.unlink(missing_ok=True)
        return {
            "video_id": video_id,
            "filename": existing["filename"],
            "stored_path": existing["stored_path"],
            "evidence_count": evidence_count,
            "status": "already_processed",
            "deduplicated": True,
        }

    if background:
        old_path = Path(existing["stored_path"]) if existing is not None else None
        temporary_path.replace(stored_path)
        with get_connection() as connection:
            upsert_video(
                connection,
                {"video_id": video_id, "filename": safe_name, "stored_path": str(stored_path),
                 "content_hash": content_hash, "status": "UPLOADED", "ocr_status": "QUEUED"},
                ["filename", "stored_path", "content_hash", "status", "ocr_status"],
            )
        if old_path is not None and old_path != stored_path:
            old_path.unlink(missing_ok=True)
        task_id = create_media_task(video_id, "TRANSCRIPTION")
        submit_task(
            task_id,
            lambda: _run_transcription_task(task_id, video_id, stored_path, safe_name),
        )
        return {
            "video_id": video_id,
            "filename": safe_name,
            "stored_path": str(stored_path),
            "status": "queued",
            "upload_task_id": task_id,
            "transcription_task_id": task_id,
            "deduplicated": False,
        }

    transcript = []
    transcription_error = ""
    try:
        transcript = await run_in_threadpool(extract_transcript_from_video, temporary_path, safe_name)
    except Exception as error:
        transcription_error = str(error)
        logger.warning("ASR failed for %s; trying OCR: %s", safe_name, error)
    ocr_evidence = []
    ocr_error = ""
    try:
        ocr_evidence = await run_in_threadpool(extract_ocr_evidence, temporary_path, safe_name)
    except Exception as error:
        # OCR is an enrichment step. Keep usable ASR, but report the exact
        # OCR failure so a missing native dependency is not invisible.
        ocr_error = str(error)
        logger.warning("OCR failed for %s; continuing with ASR evidence: %s", safe_name, error)
    if not transcript and not ocr_evidence:
        temporary_path.unlink(missing_ok=True)
        detail = "ASR and OCR produced no evidence"
        if transcription_error:
            detail += f"; ASR error: {transcription_error[:300]}"
        if ocr_error:
            detail += f"; OCR error: {ocr_error[:300]}"
        raise HTTPException(status_code=422, detail=detail)

    old_path = Path(existing["stored_path"]) if existing is not None else None
    ocr_status = (
        "DISABLED" if not ocr_enabled
        else "COMPLETED" if ocr_evidence
        else "FAILED" if ocr_error
        else "COMPLETED"
    )
    temporary_path.replace(stored_path)
    with get_connection() as connection:
        connection.execute("DELETE FROM evidence WHERE video_id = ?", (video_id,))
        evidence_rows = [
            (video_id, start, end, text, "ASR") for start, end, text in transcript
        ] + [
            (video_id, start, end, text, "OCR") for start, end, text in ocr_evidence
        ]
        connection.executemany(
            "INSERT INTO evidence(video_id, start_seconds, end_seconds, text, source) VALUES (?, ?, ?, ?, ?)",
            evidence_rows,
        )
        upsert_video(
            connection,
            {"video_id": video_id, "filename": safe_name, "stored_path": str(stored_path),
             "content_hash": content_hash, "status": "COMPLETED", "ocr_status": ocr_status,
             "transcript_text": "\n".join(item[2] for item in transcript)},
            ["filename", "stored_path", "content_hash", "status", "ocr_status", "transcript_text"],
        )
    if old_path is not None and old_path != stored_path:
        old_path.unlink(missing_ok=True)
    for path in video_dir.iterdir():
        if path.is_file() and path != stored_path and not path.name.endswith(".uploading"):
            path.unlink(missing_ok=True)
    await run_in_threadpool(sync_evidence_to_qdrant, video_id)

    return {
        "video_id": video_id,
        "filename": safe_name,
        "stored_path": str(stored_path),
        "evidence_count": len(transcript) + len(ocr_evidence),
        "asr_count": len(transcript),
        "ocr_count": len(ocr_evidence),
        **({"ocr_error": ocr_error} if ocr_error else {}),
        "status": "uploaded",
        "deduplicated": False,
    }


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    task = get_media_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    _authorize_video(str(task.get("video_id") or ""), user)
    if task.get("result_json"):
        try:
            task["result"] = json.loads(task["result_json"])
        except (TypeError, json.JSONDecodeError):
            task["result"] = None
    task.pop("result_json", None)
    return task


@app.get("/api/analysis/{task_id}")
def get_analysis_task(task_id: str, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    task = get_media_task(task_id)
    if task is None or str(task.get("task_type", "")).upper() != "ANALYSIS":
        raise HTTPException(status_code=404, detail="analysis task not found")
    _authorize_video(str(task.get("video_id") or ""), user)
    return _analysis_task_response(task)


@app.post("/api/analysis")
def create_analysis(payload: AskIn, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    """Reference-style asynchronous entry point for an initial report."""
    _authorize_video(payload.video_id, user)
    return _create_initial_analysis(payload)


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    video_id = validate_video_id(video_id)
    _authorize_video(video_id, user)
    video_dir = UPLOADS_DIR / video_id
    with get_connection() as connection:
        row = connection.execute(
            "SELECT filename FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        evidence_count = connection.execute(
            "SELECT COUNT(*) FROM evidence WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
        if row is None and evidence_count == 0 and not video_dir.exists():
            raise HTTPException(status_code=404, detail="video not found")
        connection.execute("DELETE FROM evidence WHERE video_id = ?", (video_id,))
        connection.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        connection.execute("DELETE FROM media_tasks WHERE video_id = ?", (video_id,))
        session_rows = connection.execute(
            "SELECT session_id FROM agent_sessions WHERE video_id = ?", (video_id,)
        ).fetchall()
        session_ids = [row["session_id"] for row in session_rows]
        if session_ids:
            placeholders = ",".join("?" for _ in session_ids)
            connection.execute(
                f"DELETE FROM agent_messages WHERE session_id IN ({placeholders})", session_ids
            )
            connection.execute(
                f"DELETE FROM agent_reports WHERE session_id IN ({placeholders})", session_ids
            )
            connection.execute(
                f"DELETE FROM agent_sessions WHERE session_id IN ({placeholders})", session_ids
            )
    shutil.rmtree(video_dir, ignore_errors=True)
    sync_evidence_to_qdrant(video_id)
    return {
        "video_id": video_id,
        "deleted": True,
        "filename": row["filename"] if row is not None else None,
        "evidence_count": evidence_count,
    }


@app.get("/api/videos/{video_id}/memory")
def get_video_memory(video_id: str, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    video_id = validate_video_id(video_id)
    _authorize_video(video_id, user)
    with get_connection() as connection:
        sessions = connection.execute(
            "SELECT session_id, title, summary, created_at, updated_at FROM agent_sessions "
            "WHERE video_id = ? ORDER BY updated_at DESC, session_id DESC",
            (video_id,),
        ).fetchall()
        result = []
        for session in sessions:
            messages = connection.execute(
                "SELECT role, content, created_at FROM agent_messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT 200",
                (session["session_id"],),
            ).fetchall()
            reports = connection.execute(
                "SELECT id, question, answer, answerable, support_level, report_json, created_at "
                "FROM agent_reports WHERE session_id = ? ORDER BY id DESC LIMIT 50",
                (session["session_id"],),
            ).fetchall()
            result.append({
                "session_id": session["session_id"],
                "title": session["title"],
                "summary": session["summary"] or "",
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "messages": [dict(item) for item in reversed(messages)],
                "reports": [
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "answerable": bool(item["answerable"]),
                        "support_level": item["support_level"],
                        "created_at": item["created_at"],
                        "report": parse_kimi_json(item["report_json"]) or {},
                    }
                    for item in reversed(reports)
                ],
            })
    return {
        "video_id": video_id,
        "latest_session_id": result[0]["session_id"] if result else None,
        "sessions": result,
    }


@app.get("/api/videos/{video_id}/reports")
def list_video_reports(video_id: str, limit: int = 20, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    video_id = validate_video_id(video_id)
    _authorize_video(video_id, user)
    limit = max(1, min(int(limit), 50))
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT r.id, r.session_id, r.question, r.answer, r.answerable,
                     r.support_level, r.report_json, r.report_type, r.parent_report_id, r.created_at
            FROM agent_reports r
            JOIN agent_sessions s ON s.session_id = r.session_id
            WHERE s.video_id = ? AND r.report_type = 'INITIAL'
            ORDER BY r.id DESC
            LIMIT ?
            """,
            (video_id, limit),
        ).fetchall()
    reports = []
    for row in rows:
        raw_report = parse_kimi_json(row["report_json"]) or {}
        report = raw_report.get("report") or raw_report
        citations = report.get("evidence") or report.get("citations") or []
        if not report.get("finalAnswer"):
            report = {
                **report,
                "answerable": bool(row["answerable"]),
                "title": report.get("title") or "视频分析报告",
                "finalAnswer": row["answer"],
                "evidence": citations,
                "conclusions": report.get("conclusions") or ([row["answer"]] if row["answer"] else []),
                "suggestions": report.get("suggestions") or [],
            }
        with get_connection() as connection:
            follow_up_rows = connection.execute(
                "SELECT id, question, answer, answerable, support_level, report_json, created_at "
                "FROM agent_reports WHERE parent_report_id = ? AND report_type = 'FOLLOW_UP' ORDER BY id",
                (row["id"],),
            ).fetchall()
        reports.append({
            "id": row["id"],
            "session_id": row["session_id"],
            "question": row["question"],
            "answer": row["answer"],
            "answerable": bool(row["answerable"]),
            "support_level": row["support_level"],
            "created_at": row["created_at"],
            "report": report,
            "follow_ups": [
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "answerable": bool(item["answerable"]),
                    "support_level": item["support_level"],
                    "created_at": item["created_at"],
                    "report": parse_kimi_json(item["report_json"]) or {},
                }
                for item in follow_up_rows
            ],
        })
    return {"video_id": video_id, "items": reports}


@app.delete("/api/reports/{report_id}")
def delete_report(
    report_id: int,
    user: dict[str, Any] | None = Depends(optional_user),
) -> dict[str, Any]:
    """Delete one report; deleting an initial report also deletes its replies."""
    if report_id <= 0:
        raise HTTPException(status_code=422, detail="invalid report id")
    with get_connection() as connection:
        report = connection.execute(
            "SELECT r.id, r.session_id, r.question, r.answer, r.report_type, s.video_id "
            "FROM agent_reports r JOIN agent_sessions s ON s.session_id = r.session_id WHERE r.id = ?",
            (report_id,),
        ).fetchone()
        if report is None:
            raise HTTPException(status_code=404, detail="report not found")
        _authorize_video(str(report["video_id"] or ""), user)
        is_initial = str(report["report_type"]).upper() == "INITIAL"
        follow_ups = connection.execute(
            "SELECT id, question, answer FROM agent_reports WHERE parent_report_id = ?",
            (report_id,),
        ).fetchall() if is_initial else []
        message_pairs = [("user", report["question"]), ("assistant", report["answer"])]
        message_pairs.extend(
            (role, content)
            for item in follow_ups
            for role, content in (("user", item["question"]), ("assistant", item["answer"]))
        )
        message_rows = connection.execute(
            "SELECT id, role, content FROM agent_messages WHERE session_id = ? ORDER BY id DESC",
            (report["session_id"],),
        ).fetchall()
        for role, content in message_pairs:
            matching = next(
                (item for item in message_rows if item["role"] == role and item["content"] == content),
                None,
            )
            if matching is not None:
                connection.execute("DELETE FROM agent_messages WHERE id = ?", (matching["id"],))
        if is_initial:
            connection.execute("DELETE FROM agent_reports WHERE parent_report_id = ?", (report_id,))
        connection.execute(
            "DELETE FROM agent_reports WHERE id = ?",
            (report_id,),
        )
        recent = connection.execute(
            "SELECT role, content FROM agent_messages WHERE session_id = ? ORDER BY id DESC LIMIT 12",
            (report["session_id"],),
        ).fetchall()
        summary = "\n".join(
            f"{item['role']}: {str(item['content'])[:1200]}"
            for item in reversed(recent)
        )[-6000:]
        connection.execute(
            "UPDATE agent_sessions SET summary = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (summary, report["session_id"]),
        )
    return {"report_id": report_id, "deleted": True, "deleted_children": len(follow_ups)}


@app.post("/api/ask")
def ask(payload: AskIn, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    _authorize_video(payload.video_id, user)
    _validate_analysis_video(payload.video_id)
    session_id = get_or_create_session(payload.session_id, payload.video_id)
    history = get_session_history(session_id)
    is_follow_up = bool(history)
    if kimi_is_configured():
        if is_follow_up:
            agent_result = run_agent_with_timeout(
                payload.question, payload.video_id, history, True, session_id
            )
        else:
            # Keep /api/ask compatible for existing API consumers. The web UI
            # submits initial reports through /api/analysis, the durable
            # reference-style task endpoint.
            agent_result = run_agent_with_timeout(
                payload.question, payload.video_id, history, False, session_id
            )
        if agent_result is not None:
            if agent_result.get("error"):
                # Keep provider failures as visible records rather than
                # silently dropping the submitted question.
                agent_result["session_id"] = session_id
                agent_result.setdefault("kind", "follow_up" if is_follow_up else "initial_report")
                save_agent_turn(session_id, payload.question, agent_result)
                return agent_result
            agent_result["session_id"] = session_id
            save_agent_turn(session_id, payload.question, agent_result)
            return agent_result
        result = {
            "question": payload.question,
            "answer": "DeepSeek Agent 暂时无法完成证据检索，请检查 API Key、模型名称和网络连接。",
            "grounded": False,
            "citations": [],
            "provider": "DeepSeek Agent error",
            "trace": ["DeepSeek Agent failed before submitting an accepted report"],
            "session_id": session_id,
        }
        save_agent_turn(session_id, payload.question, result)
        return result

    if is_follow_up:
        toolbox = AgentToolbox(payload.video_id, payload.question)
        selected = toolbox.evidence_for_question(payload.question, max_segments=8)
        answer = (
            "根据视频中检索到的相关内容："
            + " ".join(str(item.get("content", item.get("text", ""))).strip() for item in selected[:4])
            if selected else
            "当前视频证据不足，暂时无法确定这个追问的答案。"
        )
        result = {
            "question": payload.question,
            "answer": answer,
            "grounded": bool(selected),
            "citations": [],
            "provider": "local follow-up fallback",
            "kind": "follow_up",
            "support_level": "DIRECT" if selected else "INSUFFICIENT",
            "trace": ["Follow-up retrieval: selected question-specific evidence"],
            "session_id": session_id,
        }
        save_agent_turn(session_id, payload.question, result)
        return result

    # The old Retrieve -> Verify -> Answer graph is retained only as a
    # historical module. The active no-LLM compatibility path follows the
    # reference provider: summary questions sample the whole timeline,
    # while focused questions use targeted retrieval.
    toolbox = AgentToolbox(payload.video_id, payload.question)
    if is_summary_goal(payload.question):
        timeline = [
            item for item in toolbox.evidence
            if str(item.source).upper() != "SYSTEM"
        ]
        evidence = AgentToolbox._evenly_spaced(timeline, 18)
        trace = ["Fallback: summary goal uses representative timeline overview"]
    else:
        evidence = search_evidence(payload.question, payload.video_id)
        trace = ["Fallback: Using direct search (structured Agent unavailable)"]
    result = {
        "question": payload.question,
        "trace": trace,
        **answer_from_evidence(payload.question, evidence),
        "session_id": session_id,
    }
    save_agent_turn(session_id, payload.question, result)
    return result


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    """Get Agent performance metrics and statistics"""
    with get_connection() as connection:
        total_evidence = connection.execute(
            "SELECT COUNT(*) as count FROM evidence"
        ).fetchone()["count"]
        
        video_count = connection.execute(
            "SELECT COUNT(DISTINCT video_id) as count FROM evidence"
        ).fetchone()["count"]
        
        avg_evidence_per_video = (
            total_evidence / video_count if video_count > 0 else 0
        )
    
    qdrant_active = get_qdrant_client() is not None
    embedding_backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip().lower()
    embedding_model = (
        os.getenv("EMBEDDING_MODEL_REPO", "Xenova/bge-small-zh-v1.5")
        if embedding_backend in {"onnx", "bge-onnx", "local-onnx"}
        else os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )
    embedding_enabled = env_flag("EMBEDDING_ENABLED", False)
    embedding_available = embedding_enabled and get_embedding_model() is not None
    
    return {
        "agent_version": "0.5.0",
        "capabilities": [
            "vector_retrieval",
            "keyword_retrieval",
            "evidence_verification",
            "answer_grounding",
            "citation_tracking",
            "agent_tracing",
            "tool_calling",
            "structured_planning",
            "critic_revision",
        ],
        "statistics": {
            "total_evidence": total_evidence,
            "distinct_videos": video_count,
            "avg_evidence_per_video": round(avg_evidence_per_video, 2),
        },
        "components": {
            "vector_database": "qdrant" if qdrant_active and embedding_available else "disabled",
            "embedding_model": embedding_model if embedding_available else "unavailable",
            "agent_framework": "structured-langgraph" if kimi_is_configured() else "deterministic-fallback",
            "agent_workflow": os.getenv("DEEPSEEK_AGENT_WORKFLOW", os.getenv("KIMI_AGENT_WORKFLOW", "structured")),
            "transcription": "faster-whisper" if ocr_runner._WHISPER_MODEL is not None else "available on upload",
            "video_processing": "ffmpeg" if resolve_ffmpeg_path() else "unavailable",
            "llm": {
                "provider": "DeepSeek" if kimi_is_configured() else "local fallback",
                "configured": kimi_is_configured(),
                "model": llm_settings()["model"],
            },
        },
        "available_tools": AGENT_TOOL_NAMES,
    }
