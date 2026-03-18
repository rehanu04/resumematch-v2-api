from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import json5
import numpy as np
import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

load_dotenv()

LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "gemini")
LM_BASE_URL    = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
LM_MODEL       = os.getenv("LMSTUDIO_MODEL", "google/gemma-3-4b")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
DATABASE_URL   = os.getenv("DATABASE_URL")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

app = FastAPI(title="AI Interview Coach (Cloud)", version="0.8.0")

_WHISPER = None


# ---------------------------
# Helpers
# ---------------------------

def sanitize_text(s: str) -> str:
    if "â" in s or "Ã" in s:
        try:
            s = s.encode("latin-1").decode("utf-8")
        except Exception:
            pass
    s = (
        s.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2026", "...")
    )
    s = s.replace("â", "'")
    return s


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_json_tolerant(text: str) -> Dict[str, Any]:
    cleaned = sanitize_text(strip_code_fences(text))
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        return json5.loads(cleaned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expected JSON but got invalid output. Error: {e}. Raw: {cleaned[:400]}")


def filler_stats(text: str) -> Tuple[int, Dict[str, int]]:
    lower = text.lower()
    fillers = ["um", "uh", "like", "you know", "actually", "basically", "so"]
    counts = {f: lower.count(f) for f in fillers}
    return sum(counts.values()), counts


def base_interview_system(role_title: str | None, job_description: str | None) -> str:
    role_part = f"Role: {role_title}\n" if role_title else ""
    jd_part = f"Job Description:\n{job_description}\n" if job_description else ""
    return (
        "You are an AI interviewer.\n"
        "Rules:\n"
        "1) Ask ONE question at a time.\n"
        "2) Use provided candidate facts when relevant. Do NOT invent candidate history.\n"
        "3) Ask targeted follow-ups when the answer is vague (metrics, scope, trade-offs, edge cases).\n"
        "4) Keep it realistic and aligned to the job.\n"
        "5) Use plain punctuation (use ' and \").\n\n"
        f"{role_part}{jd_part}"
    )


# ---------------------------
# DB (PostgreSQL via psycopg2)
# ---------------------------

def db_connect():
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL not set in environment.")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


def db_init() -> None:
    if not DATABASE_URL:
        return  # Skip init if no DB URL (e.g., during build phase)
    
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                  id TEXT PRIMARY KEY,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  profile_json TEXT NOT NULL,
                  raw_text TEXT
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                  id SERIAL PRIMARY KEY,
                  profile_id TEXT NOT NULL REFERENCES profiles(id),
                  chunk_text TEXT NOT NULL,
                  embedding BYTEA NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_profile_id ON chunks(profile_id);")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS drills (
                  id TEXT PRIMARY KEY,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  mode TEXT NOT NULL,
                  question TEXT,
                  transcript TEXT NOT NULL,
                  coach_json TEXT NOT NULL,
                  filler_json TEXT NOT NULL,
                  segments_json TEXT
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_turns (
                  id TEXT PRIMARY KEY,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  session_id TEXT NOT NULL,
                  mode TEXT NOT NULL,
                  question TEXT NOT NULL,
                  transcript TEXT NOT NULL,
                  next_question TEXT NOT NULL,
                  score_json TEXT,
                  rewrites_json TEXT,
                  speech_coach_json TEXT,
                  fillers_json TEXT,
                  segments_json TEXT
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                  session_id TEXT PRIMARY KEY,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  profile_id TEXT,
                  messages_json TEXT NOT NULL
                );
                """
            )
    finally:
        conn.close()


@app.on_event("startup")
def _startup() -> None:
    db_init()


# ---------------------------
# LLM + Embeddings
# ---------------------------

async def call_llm(messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 600) -> Dict[str, Any]:
    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in .env")
        gemini_messages = []
        system_text = ""
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [{"text": m["content"]}]})
            elif m["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [{"text": m["content"]}]})
        payload: Dict[str, Any] = {
            "contents": gemini_messages,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        url = f"[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/){GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"choices": [{"message": {"content": text}}]}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Gemini error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini not reachable: {e}")
    else:
        payload = {"model": LM_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": False}
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                r = await client.post(f"{LM_BASE_URL}/chat/completions", json=payload)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"LM Studio error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LM Studio not reachable: {e}")


async def embed_texts(texts: List[str]) -> List[np.ndarray]:
    # Using LM Studio for embeddings in this example. If moving fully to cloud, 
    # you may want to integrate Gemini's embedding models here instead.
    payload = {"model": EMBED_MODEL, "input": texts}
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(f"{LM_BASE_URL}/embeddings", json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Embedding error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding service not reachable: {e}")

    vecs: List[np.ndarray] = []
    for item in data.get("data", []):
        emb = item.get("embedding")
        if not isinstance(emb, list):
            raise HTTPException(status_code=500, detail="Invalid embedding response format")
        vecs.append(np.array(emb, dtype=np.float32))
    if len(vecs) != len(texts):
        raise HTTPException(status_code=500, detail="Embedding count mismatch")
    return vecs


# ---------------------------
# RAG
# ---------------------------

def chunk_text(text: str, max_chars: int = 600, overlap: int = 80) -> List[str]:
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(t):
        end = min(len(t), start + max_chars)
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(t):
            break
        start = max(0, end - overlap)
    return chunks


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def ensure_profile_has_chunks(profile_id: str) -> None:
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE profile_id = %s", (profile_id,))
            row = cur.fetchone()
            if not row or int(row[0]) == 0:
                raise HTTPException(status_code=400, detail="Profile has no indexed chunks. Run /rag/index first.")
    finally:
        conn.close()


async def rag_retrieve_context(profile_id: str, query: str, top_k: int = 4) -> str:
    q_vec = (await embed_texts([query]))[0]
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_text, embedding FROM chunks WHERE profile_id = %s", (profile_id,))
            rows = cur.fetchall()
            
            if not rows:
                raise HTTPException(status_code=400, detail="Profile has no indexed chunks. Run /rag/index first.")

            scored: List[Tuple[float, str]] = []
            for chunk_text_val, emb_blob in rows:
                vec = np.frombuffer(emb_blob, dtype=np.float32)
                scored.append((cosine_sim(q_vec, vec), str(chunk_text_val)))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:top_k]
            bullets = "\n".join([f"- {t}" for _, t in top])
            return f"Allowed candidate facts (ground truth):\n{bullets}\n"
    finally:
        conn.close()


# ---------------------------
# Coaching
# ---------------------------

async def score_answer_llm(interview_context: str, question: str, answer: str, rag_context: str) -> Dict[str, Any]:
    sys = (
        "You are a strict technical interviewer scoring a candidate answer.\n"
        "Return ONLY valid JSON. No markdown.\n"
        "Rubric (0-5 each): relevance, clarity, depth, evidence.\n"
        "Also overall (0-100), strengths (3), improvements (3).\n"
        "If the answer claims facts not supported by allowed candidate facts, set hallucination_risk=true.\n"
        "Schema:\n"
        "{\n"
        '  \"relevance\": number,\n'
        '  \"clarity\": number,\n'
        '  \"depth\": number,\n'
        '  \"evidence\": number,\n'
        '  \"overall\": number,\n'
        '  \"hallucination_risk\": boolean,\n'
        '  \"strengths\": [string],\n'
        '  \"improvements\": [string]\n'
        "}\n"
    )
    user = (
        f"Interview context:\n{interview_context}\n\n"
        f"{rag_context}\n"
        f"Question:\n{question}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Return JSON only."
    )
    data = await call_llm([{"role": "system", "content": sys}, {"role": "user", "content": user}], temperature=0.0, max_tokens=450)
    return parse_json_tolerant(data["choices"][0]["message"]["content"])


async def rewrite_answer_llm(interview_context: str, question: str, answer: str, rag_context: str) -> Dict[str, Any]:
    sys = (
        "You are an interview coach.\n"
        "Return ONLY valid JSON. No markdown.\n"
        "TASK: Rewrite the candidate answer into improved answers.\n"
        "CRITICAL RULES:\n"
        "1) Outputs must be ANSWERS in first person (I...). Do NOT ask questions.\n"
        "2) You may ONLY use facts explicitly present in:\n"
        "   - the candidate answer text, OR\n"
        "   - the allowed candidate facts.\n"
        "3) If allowed facts list a skill, treat it as a skill. Do NOT claim you used it unless explicitly stated.\n"
        "4) Do NOT add mechanisms (DB, algorithms, caching, async, logging) unless explicitly present.\n"
        "5) Do NOT add performance numbers. Use [add metric].\n"
        "6) If missing details, use [add detail].\n"
        "Schema:\n"
        "{\n"
        '  \"simple_clear\": string,\n'
        '  \"interview_strong\": string,\n'
        '  \"tips\": [string]\n'
        "}\n"
    )
    user = (
        f"Interview context:\n{interview_context}\n\n"
        f"{rag_context}\n"
        f"Question:\n{question}\n\n"
        f"Original candidate answer:\n{answer}\n\n"
        "Return JSON only."
    )
    data = await call_llm([{"role": "system", "content": sys}, {"role": "user", "content": user}], temperature=0.1, max_tokens=650)
    return parse_json_tolerant(data["choices"][0]["message"]["content"])


async def speech_coach_llm(question: str, transcript: str, rag_context: str) -> Dict[str, Any]:
    sys = (
        "You are an English communication coach for interview speaking.\n"
        "Return ONLY valid JSON. No markdown.\n"
        "CRITICAL RULES:\n"
        "1) Improve ONLY the answer transcript into a better ANSWER. Do NOT restate the question.\n"
        "2) Do NOT ask questions. Write improved answers as statements in first person (I...).\n"
        "3) Do NOT invent new content. Use ONLY transcript + allowed candidate facts.\n"
        "4) If transcript does not answer the question, provide a short suggested answer grounded in allowed facts with [add detail].\n"
        "Schema:\n"
        "{\n"
        '  \"grammar_fixes\": [{\"from\":string,\"to\":string}],\n'
        '  \"clarity_score\": number,\n'
        '  \"structure_score\": number,\n'
        '  \"improved_simple\": string,\n'
        '  \"improved_interview\": string,\n'
        '  \"next_drills\": [string]\n'
        "}\n"
    )
    user = (
        f"Question:\n{question}\n\n"
        f"{rag_context}\n"
        f"Answer transcript:\n{transcript}\n\n"
        "Return JSON only."
    )
    data = await call_llm([{"role": "system", "content": sys}, {"role": "user", "content": user}], temperature=0.1, max_tokens=700)
    return parse_json_tolerant(data["choices"][0]["message"]["content"])


# ---------------------------
# Whisper
# ---------------------------

def _get_whisper():
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel
        _WHISPER = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    return _WHISPER


async def transcribe_upload(file: UploadFile) -> Tuple[str, List[Dict[str, Any]]]:
    tmp_dir = Path("data") / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ext = ""
    if file.filename and "." in file.filename:
        ext = "." + file.filename.split(".")[-1].lower()
    if ext not in [".wav", ".m4a", ".mp3", ".ogg", ".webm", ".flac", ".aac"]:
        ext = ".bin"

    tmp_path = tmp_dir / f"{uuid.uuid4().hex}{ext}"
    tmp_path.write_bytes(await file.read())

    try:
        model = _get_whisper()
        segments, _info = model.transcribe(str(tmp_path), vad_filter=True)
        segs: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        for s in segments:
            segs.append({"start": float(s.start), "end": float(s.end), "text": s.text})
            text_parts.append(s.text)
        transcript = " ".join([t.strip() for t in text_parts]).strip()
        return transcript, segs
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------
# Request models
# ---------------------------

class ProfileIngestRequest(BaseModel):
    raw_text: str = Field(..., min_length=50)
    name_hint: Optional[str] = None


class ProfileSaveRequest(BaseModel):
    profile: Dict[str, Any]
    raw_text: Optional[str] = None


class RagIndexRequest(BaseModel):
    profile_id: str
    max_chunk_chars: int = Field(default=600, ge=200, le=2000)
    overlap: int = Field(default=80, ge=0, le=400)


class RagSearchRequest(BaseModel):
    profile_id: str
    query: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)


class SessionStartRequest(BaseModel):
    role_title: str | None = None
    job_description: str | None = None
    profile_id: str | None = None


class SessionTurnRequest(BaseModel):
    session_id: str = Field(..., min_length=6)
    answer: str = Field(..., min_length=1)
    do_score: bool = True
    do_rewrite: bool = True


class SpeechFeedbackRequest(BaseModel):
    text: str = Field(..., min_length=5)
    mode: str = Field(default="general")  


class InterviewVoiceStartRequest(BaseModel):
    role_title: str | None = None
    job_description: str | None = None
    profile_id: str | None = None


# ---------------------------
# Endpoints
# ---------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/profile/ingest")
async def profile_ingest(req: ProfileIngestRequest) -> Dict[str, Any]:
    name_line = f"Name hint: {req.name_hint}\n" if req.name_hint else ""
    sys = (
        "You extract a candidate profile from raw resume/portfolio text.\n"
        "Return ONLY valid JSON. No markdown.\n"
        "Keys: name, headline, summary, skills(list), projects(list), experience(list), education(list), links(list).\n"
        "Rules: if unknown use empty string/list; deduplicate skills.\n"
    )
    user = f"{name_line}RAW TEXT:\n{req.raw_text}\n\nReturn JSON only."
    data = await call_llm([{"role": "system", "content": sys}, {"role": "user", "content": user}], temperature=0.0, max_tokens=900)
    return {"profile": parse_json_tolerant(data["choices"][0]["message"]["content"]), "model": LM_MODEL}


@app.post("/profile/ingest_file")
async def profile_ingest_file(file: UploadFile = File(...), name_hint: Optional[str] = None) -> Dict[str, Any]:
    content = (await file.read()).decode("utf-8", errors="ignore")
    return await profile_ingest(ProfileIngestRequest(raw_text=content, name_hint=name_hint))


@app.post("/profile/save")
def profile_save(req: ProfileSaveRequest) -> Dict[str, Any]:
    profile_id = uuid.uuid4().hex[:12]
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO profiles (id, profile_json, raw_text) VALUES (%s, %s, %s)",
                (profile_id, json.dumps(req.profile, ensure_ascii=False), req.raw_text),
            )
    finally:
        conn.close()
    return {"profile_id": profile_id}


@app.post("/rag/index")
async def rag_index(req: RagIndexRequest) -> Dict[str, Any]:
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT profile_json, raw_text FROM profiles WHERE id = %s", (req.profile_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Profile not found")
            
            raw_text = row[1] or ""
            text = raw_text.strip() if raw_text.strip() else row[0]
            chunks = chunk_text(text, max_chars=req.max_chunk_chars, overlap=req.overlap)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="Nothing to index")
            
            vecs = await embed_texts(chunks)

            cur.execute("DELETE FROM chunks WHERE profile_id = %s", (req.profile_id,))
            
            for chunk, vec in zip(chunks, vecs):
                cur.execute(
                    "INSERT INTO chunks (profile_id, chunk_text, embedding) VALUES (%s, %s, %s)",
                    (req.profile_id, chunk, psycopg2.Binary(vec.astype(np.float32).tobytes())),
                )
            
            return {"profile_id": req.profile_id, "chunks_indexed": len(chunks), "embed_model": EMBED_MODEL}
    finally:
        conn.close()


@app.post("/rag/search")
async def rag_search(req: RagSearchRequest) -> Dict[str, Any]:
    ensure_profile_has_chunks(req.profile_id)
    q_vec = (await embed_texts([req.query]))[0]
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, chunk_text, embedding FROM chunks WHERE profile_id = %s", (req.profile_id,))
            rows = cur.fetchall()
            
            scored: List[Tuple[float, int, str]] = []
            for cid, ctext, blob in rows:
                vec = np.frombuffer(blob, dtype=np.float32)
                scored.append((cosine_sim(q_vec, vec), int(cid), str(ctext)))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[: req.top_k]
            return {"profile_id": req.profile_id, "query": req.query, "top_k": req.top_k,
                    "results": [{"chunk_id": cid, "score": float(score), "text": text} for (score, cid, text) in top]}
    finally:
        conn.close()


@app.post("/session/start")
async def session_start(req: SessionStartRequest) -> Dict[str, Any]:
    session_id = uuid.uuid4().hex[:12]
    system = base_interview_system(req.role_title, req.job_description)

    context = ""
    if req.profile_id:
        ensure_profile_has_chunks(req.profile_id)
        context = await rag_retrieve_context(req.profile_id, "Summarize candidate background, projects, skills", top_k=4)

    kickoff = {"role": "user", "content": (context + "\nStart the interview now. Ask your first question only. Then wait.").strip()}
    data = await call_llm([{"role": "system", "content": system}, kickoff], temperature=0.3, max_tokens=400)
    first_q = sanitize_text(data["choices"][0]["message"]["content"])

    messages = [{"role": "system", "content": system}, kickoff, {"role": "assistant", "content": first_q}]

    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (session_id, profile_id, messages_json) VALUES (%s, %s, %s)",
                (session_id, req.profile_id, json.dumps(messages, ensure_ascii=False)),
            )
    finally:
        conn.close()

    return {"session_id": session_id, "question": first_q, "model": LM_MODEL if LLM_PROVIDER == "lmstudio" else GEMINI_MODEL}


@app.post("/session/turn")
async def session_turn(req: SessionTurnRequest) -> Dict[str, Any]:
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT messages_json, profile_id FROM sessions WHERE session_id = %s", (req.session_id,)
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Session not found. Call /session/start or /interview_voice/start first.")

    messages: List[Dict[str, str]] = json.loads(row[0])
    profile_id: Optional[str] = row[1]

    last_question = ""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_question = str(m.get("content", "")).strip()
            break
    if not last_question:
        last_question = "Interview question"

    rag_context = ""
    if profile_id:
        ensure_profile_has_chunks(profile_id)
        rag_context = await rag_retrieve_context(profile_id, req.answer, top_k=4)

    user_payload = (
        f"{rag_context}\n"
        f"Candidate answer:\n{req.answer}\n\n"
        "Now ask the next interview question (or a follow-up) only."
    ).strip()

    messages.append({"role": "user", "content": user_payload})
    data = await call_llm(messages, temperature=0.3, max_tokens=400)
    next_q = sanitize_text(data["choices"][0]["message"]["content"])
    messages.append({"role": "assistant", "content": next_q})

    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET messages_json = %s WHERE session_id = %s",
                (json.dumps(messages, ensure_ascii=False), req.session_id),
            )
    finally:
        conn.close()

    system_prompt = str(messages[0].get("content", ""))
    score_obj = await score_answer_llm(system_prompt, last_question, req.answer, rag_context) if req.do_score else None
    rewrites_obj = await rewrite_answer_llm(system_prompt, last_question, req.answer, rag_context) if req.do_rewrite else None

    return {"session_id": req.session_id, "question": next_q, "model": LM_MODEL if LLM_PROVIDER == "lmstudio" else GEMINI_MODEL, "score": score_obj, "rewrites": rewrites_obj}


@app.post("/speech/transcribe")
async def speech_transcribe(file: UploadFile = File(...)) -> Dict[str, Any]:
    transcript, segs = await transcribe_upload(file)
    return {"text": transcript, "segments": segs, "whisper_model": WHISPER_MODEL}


@app.post("/speech/feedback")
async def speech_feedback(req: SpeechFeedbackRequest) -> Dict[str, Any]:
    t = req.text.strip()
    total, counts = filler_stats(t)

    sys = (
        "You are an English communication coach.\n"
        "Return ONLY valid JSON. No markdown.\n"
        "Do NOT invent content; only improve the provided text.\n"
        "Outputs must be statements (no questions).\n"
        "Schema:\n"
        "{\n"
        '  \"grammar_fixes\": [{\"from\":string,\"to\":string}],\n'
        '  \"clarity_score\": number,\n'
        '  \"structure_score\": number,\n'
        '  \"improved_simple\": string,\n'
        '  \"improved_interview\": string,\n'
        '  \"next_drills\": [string]\n'
        "}\n"
    )
    user = f"Mode: {req.mode}\nText:\n{t}\n\nReturn JSON only."
    data = await call_llm([{"role": "system", "content": sys}, {"role": "user", "content": user}], temperature=0.1, max_tokens=700)
    coach = parse_json_tolerant(data["choices"][0]["message"]["content"])

    return {"input_text": t, "fillers_total": total, "filler_counts": counts, "coach": coach}


@app.post("/drill/run")
async def drill_run(file: UploadFile = File(...), mode: str = Form("general"), question: str = Form("")) -> Dict[str, Any]:
    transcript, segs = await transcribe_upload(file)
    total, counts = filler_stats(transcript)

    rag_context = "Allowed facts:\n- Use only the transcript.\n"
    coach = await speech_coach_llm(question or "(general speaking)", transcript, rag_context)

    drill_id = uuid.uuid4().hex[:12]
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO drills (id, mode, question, transcript, coach_json, filler_json, segments_json) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    drill_id,
                    mode,
                    question,
                    transcript,
                    json.dumps(coach, ensure_ascii=False),
                    json.dumps({"fillers_total": total, "filler_counts": counts}, ensure_ascii=False),
                    json.dumps(segs, ensure_ascii=False),
                ),
            )
    finally:
        conn.close()

    return {"drill_id": drill_id, "mode": mode, "question": question, "transcript": transcript,
            "fillers_total": total, "filler_counts": counts, "coach": coach, "segments": segs}


@app.get("/drill/history")
def drill_history(limit: int = 20) -> Dict[str, Any]:
    limit = max(1, min(100, int(limit)))
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, created_at, mode, question, transcript, coach_json, filler_json FROM drills ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    items = []
    for rid, created_at, mode, question, transcript, coach_json, filler_json in rows:
        items.append({
            "drill_id": rid,
            "created_at": str(created_at),
            "mode": mode,
            "question": question,
            "transcript": transcript,
            "coach": json.loads(coach_json),
            "fillers": json.loads(filler_json),
        })
    return {"count": len(items), "items": items}


@app.post("/interview_voice/start")
async def interview_voice_start(req: InterviewVoiceStartRequest) -> Dict[str, Any]:
    return await session_start(SessionStartRequest(role_title=req.role_title, job_description=req.job_description, profile_id=req.profile_id))


@app.post("/interview_voice/turn")
async def interview_voice_turn(session_id: str = Form(...), file: UploadFile = File(...)) -> Dict[str, Any]:
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT messages_json, profile_id FROM sessions WHERE session_id = %s", (session_id,)
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Session not found. Call /interview_voice/start first.")

    messages: List[Dict[str, str]] = json.loads(row[0])
    profile_id: Optional[str] = row[1]

    last_question = ""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_question = str(m.get("content", "")).strip()
            break
    if not last_question:
        last_question = "Interview question"

    transcript, segs = await transcribe_upload(file)
    total, counts = filler_stats(transcript)

    rag_context = ""
    if profile_id:
        ensure_profile_has_chunks(profile_id)
        rag_context = await rag_retrieve_context(profile_id, last_question + " " + transcript, top_k=4)

    speech_coach = await speech_coach_llm(last_question, transcript, rag_context)

    turn_out = await session_turn(SessionTurnRequest(session_id=session_id, answer=transcript, do_score=True, do_rewrite=True))

    turn_id = uuid.uuid4().hex[:12]
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO interview_turns (id, session_id, mode, question, transcript, next_question, score_json, rewrites_json, speech_coach_json, fillers_json, segments_json) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    turn_id,
                    session_id,
                    "interview",
                    last_question,
                    transcript,
                    turn_out.get("question", ""),
                    json.dumps(turn_out.get("score"), ensure_ascii=False),
                    json.dumps(turn_out.get("rewrites"), ensure_ascii=False),
                    json.dumps(speech_coach, ensure_ascii=False),
                    json.dumps({"fillers_total": total, "filler_counts": counts}, ensure_ascii=False),
                    json.dumps(segs, ensure_ascii=False),
                ),
            )
    finally:
        conn.close()

    return {
        "turn_id": turn_id,
        "session_id": session_id,
        "question": last_question,
        "transcript": transcript,
        "fillers_total": total,
        "filler_counts": counts,
        "speech_coach": speech_coach,
        "next_question": turn_out.get("question"),
        "score": turn_out.get("score"),
        "rewrites": turn_out.get("rewrites"),
        "segments": segs,
    }


@app.get("/interview_voice/history")
def interview_voice_history(session_id: str, limit: int = 50) -> Dict[str, Any]:
    limit = max(1, min(200, int(limit)))
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, created_at, question, transcript, next_question, score_json, rewrites_json, speech_coach_json, fillers_json "
                "FROM interview_turns WHERE session_id = %s ORDER BY created_at ASC LIMIT %s",
                (session_id, limit),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    items = []
    for rid, created_at, q, tr, nq, score_json, rewrites_json, speech_coach_json, fillers_json in rows:
        items.append({
            "turn_id": rid,
            "created_at": str(created_at),
            "question": q,
            "transcript": tr,
            "next_question": nq,
            "score": json.loads(score_json) if score_json else None,
            "rewrites": json.loads(rewrites_json) if rewrites_json else None,
            "speech_coach": json.loads(speech_coach_json) if speech_coach_json else None,
            "fillers": json.loads(fillers_json) if fillers_json else None,
        })

    return {"session_id": session_id, "count": len(items), "items": items}
