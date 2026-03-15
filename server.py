from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import json
import os
import sys
import asyncio
import base64
import logging
import re
import time
import math
import httpx

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from google import genai

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('sub_extractor')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html", media_type="text/html")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────
#  Prompt template for Gemini time-range transcription
# ──────────────────────────────────────────────────────────
def make_gemini_prompt(start_mm_ss=None, end_mm_ss=None):
    time_range = ""
    if start_mm_ss and end_mm_ss:
        time_range = f"\nTRANSCRIBE ONLY the section from {start_mm_ss} to {end_mm_ss}. Ignore audio outside this range."

    return f"""You are a professional subtitle transcription engine. Transcribe ALL spoken words in this audio with precise timestamps.

CRITICAL RULES:
1. Auto-detect the spoken language. Output transcription in the ORIGINAL spoken language (do NOT translate).
2. Each subtitle segment should be 1-2 sentences, roughly 3-8 seconds long.
3. Timestamps MUST be precise — align with actual speech start/end.
4. Include ALL spoken content in the specified range, do not skip anything.
5. Output ONLY valid JSON array, no other text.{time_range}

OUTPUT FORMAT (strict JSON):
[
  {{"index": 1, "start": "00:00:01,200", "end": "00:00:04,800", "text": "text here"}},
  {{"index": 2, "start": "00:00:05,100", "end": "00:00:08,300", "text": "next segment"}}
]

TIMESTAMP FORMAT: HH:MM:SS,mmm (hours:minutes:seconds,milliseconds)
- Use commas for milliseconds separator (SRT standard)
- Be precise — start when speech begins, end when it stops

IMPORTANT: Output ONLY the JSON array. No markdown, no code fences, no explanation."""


# ──────────────────────────────────────────────────────────
#  Parsing helpers
# ──────────────────────────────────────────────────────────
def parse_srt_json(text: str) -> list:
    """Parse Gemini response into list of subtitle entries."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
    text = text.strip()

    try:
        entries = json.loads(text)
        if isinstance(entries, list):
            return entries
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            entries = json.loads(match.group())
            if isinstance(entries, list):
                return entries
        except json.JSONDecodeError:
            pass
    return []


def format_timestamp(ts: str) -> str:
    """Normalize timestamp to SRT format HH:MM:SS,mmm"""
    ts = ts.strip()
    if re.match(r'^\d{2}:\d{2}:\d{2},\d{3}$', ts):
        return ts
    ts = ts.replace('.', ',')
    if re.match(r'^\d{2}:\d{2}:\d{2},\d{3}$', ts):
        return ts
    if re.match(r'^\d{2}:\d{2},\d{3}$', ts):
        return "00:" + ts
    if re.match(r'^\d{2}:\d{2}$', ts):
        return "00:" + ts + ",000"
    if re.match(r'^\d{2}:\d{2}:\d{2}$', ts):
        return ts + ",000"
    try:
        secs = float(ts.replace(',', '.'))
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        ms = int((secs % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    except:
        return ts


def ts_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS,mmm to seconds."""
    ts = format_timestamp(ts)
    try:
        parts = ts.replace(',', '.').split(':')
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except:
        return 0.0


def seconds_to_mmss(secs: float) -> str:
    """Convert seconds to MM:SS format for Gemini prompt."""
    m = int(secs // 60)
    s = int(secs % 60)
    return f"{m:02d}:{s:02d}"


def entries_to_srt(entries: list) -> str:
    """Convert entries to SRT format string."""
    lines = []
    for i, entry in enumerate(entries, 1):
        start = format_timestamp(str(entry.get("start", "00:00:00,000")))
        end = format_timestamp(str(entry.get("end", "00:00:00,000")))
        text = entry.get("text", "").strip()
        if text:
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
    return "\n".join(lines)


def detect_language_from_entries(entries: list) -> str:
    if not entries:
        return "unknown"
    text = " ".join(e.get("text", "") for e in entries[:5])
    vn_chars = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")
    if any(c in vn_chars for c in text.lower()):
        return "Tiếng Việt"
    if re.search(r'[\u4e00-\u9fff]', text):
        return "中文"
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
        return "日本語"
    if re.search(r'[\uac00-\ud7af]', text):
        return "한국어"
    if re.search(r'[\u0e00-\u0e7f]', text):
        return "ภาษาไทย"
    return "English"


MIME_MAP = {
    '.mp3': 'audio/mp3', '.wav': 'audio/wav', '.aac': 'audio/aac',
    '.ogg': 'audio/ogg', '.flac': 'audio/flac', '.aiff': 'audio/aiff',
    '.m4a': 'audio/mp4', '.mp4': 'video/mp4', '.mkv': 'video/x-matroska',
    '.webm': 'video/webm', '.mov': 'video/quicktime',
}


# ──────────────────────────────────────────────────────────
#  Parse tagged keys: "gemini:AIza..." or "groq:gsk_..."
# ──────────────────────────────────────────────────────────
def parse_keys(raw_keys: list) -> list:
    """Parse keys into [{provider, key, model}] format.
    Auto-detect provider from key prefix if no tag given.
    """
    parsed = []
    for raw in raw_keys:
        raw = raw.strip()
        if not raw:
            continue

        provider = None
        key = raw

        # Check for explicit tag
        if ':' in raw and not raw.startswith('AIza'):
            tag, k = raw.split(':', 1)
            tag = tag.lower().strip()
            key = k.strip()
            if tag in ('gemini', 'gem', 'g'):
                provider = 'gemini'
            elif tag in ('groq', 'grq', 'q'):
                provider = 'groq'

        # Auto-detect from key prefix
        if not provider:
            if key.startswith('AIza'):
                provider = 'gemini'
            elif key.startswith('gsk_'):
                provider = 'groq'
            else:
                provider = 'gemini'  # default

        parsed.append({"provider": provider, "key": key})

    return parsed


# ──────────────────────────────────────────────────────────
#  Groq Whisper transcription
# ──────────────────────────────────────────────────────────
async def groq_transcribe(file_path: str, api_key: str, model: str = "whisper-large-v3-turbo") -> list:
    """Transcribe audio using Groq Whisper API. Returns list of entries."""
    mime = MIME_MAP.get(os.path.splitext(file_path)[1].lower(), 'audio/mp3')
    fname = os.path.basename(file_path)

    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(file_path, "rb") as f:
            resp = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                data={
                    "model": model,
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                },
                files={"file": (fname, f, mime)},
            )

    if resp.status_code != 200:
        raise Exception(f"Groq API error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    entries = []
    segments = data.get("segments", [])
    for i, seg in enumerate(segments, 1):
        start_s = seg.get("start", 0)
        end_s = seg.get("end", 0)
        text = seg.get("text", "").strip()
        if text:
            h1, m1, s1 = int(start_s // 3600), int((start_s % 3600) // 60), start_s % 60
            h2, m2, s2 = int(end_s // 3600), int((end_s % 3600) // 60), end_s % 60
            entries.append({
                "index": i,
                "start": f"{h1:02d}:{m1:02d}:{int(s1):02d},{int((s1 % 1) * 1000):03d}",
                "end": f"{h2:02d}:{m2:02d}:{int(s2):02d},{int((s2 % 1) * 1000):03d}",
                "text": text
            })
    return entries


# ──────────────────────────────────────────────────────────
#  Gemini transcription (single chunk or full)
# ──────────────────────────────────────────────────────────
async def gemini_transcribe_chunk(client, uploaded_file, model: str,
                                   start_mm_ss=None, end_mm_ss=None) -> list:
    """Transcribe a time range of audio using Gemini."""
    prompt = make_gemini_prompt(start_mm_ss, end_mm_ss)
    loop = asyncio.get_event_loop()

    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=model,
            contents=[prompt, uploaded_file],
            config={"temperature": 0.1, "max_output_tokens": 65536}
        )
    )
    return parse_srt_json(response.text)


# ──────────────────────────────────────────────────────────
#  WebSocket endpoint — main transcription handler
# ──────────────────────────────────────────────────────────
@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    await ws.accept()
    session_id = f"ws_{int(time.time() * 1000)}"
    temp_path = None
    logger.info(f"[WS] Session {session_id}: connected")

    try:
        raw = await ws.receive_text()
        config = json.loads(raw)

        file_data_b64 = config.get("file_data", "")
        file_name = config.get("file_name", "audio.mp3")
        api_keys_raw = config.get("api_keys", [])
        model_name = config.get("model_name", "gemini-2.5-flash")
        groq_model = config.get("groq_model", "whisper-large-v3-turbo")
        duration_hint = config.get("duration_hint", 0)
        threads_per_key = max(1, min(20, config.get("threads_per_key", 5)))

        if not file_data_b64:
            await ws.send_json({"type": "error", "message": "No file data received"})
            return
        if not api_keys_raw:
            await ws.send_json({"type": "error", "message": "No API keys provided"})
            return

        # Decode and save file
        file_bytes = base64.b64decode(file_data_b64)
        file_size_mb = len(file_bytes) / (1024 * 1024)
        ext = os.path.splitext(file_name)[1].lower()

        temp_path = os.path.join(TEMP_DIR, f"{session_id}_{file_name}")
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        await ws.send_json({
            "type": "progress", "step": "upload",
            "message": f"📁 File: {file_name} ({file_size_mb:.1f} MB)", "percent": 5
        })

        # Parse keys
        parsed_keys = parse_keys(api_keys_raw)
        gemini_keys = [k for k in parsed_keys if k["provider"] == "gemini"]
        groq_keys = [k for k in parsed_keys if k["provider"] == "groq"]
        total_workers = len(gemini_keys) * threads_per_key + len(groq_keys)

        await ws.send_json({
            "type": "progress", "step": "keys",
            "message": f"🔑 {len(gemini_keys)} Gemini (×{threads_per_key}={len(gemini_keys)*threads_per_key}) + {len(groq_keys)} Groq = {total_workers} workers",
            "percent": 8
        })

        all_entries = []
        errors = []

        # ─── GROQ WORKERS ───
        if groq_keys:
            await ws.send_json({
                "type": "progress", "step": "groq",
                "message": f"⚡ Groq Whisper: {len(groq_keys)} worker(s) đang chạy...",
                "percent": 15
            })

            # Groq Whisper: send full file to each key (they're fast, ~10x realtime)
            # Use first key that succeeds
            groq_done = False
            for i, gk in enumerate(groq_keys):
                if groq_done:
                    break
                try:
                    await ws.send_json({
                        "type": "worker", "provider": "groq", "worker_id": i + 1,
                        "status": "running", "message": f"Groq #{i+1} đang transcribe..."
                    })
                    entries = await groq_transcribe(temp_path, gk["key"], groq_model)
                    if entries:
                        all_entries.extend(entries)
                        groq_done = True
                        await ws.send_json({
                            "type": "worker", "provider": "groq", "worker_id": i + 1,
                            "status": "done", "message": f"Groq #{i+1}: {len(entries)} segments ✅"
                        })
                        # Stream partial results immediately
                        await ws.send_json({
                            "type": "partial",
                            "entries": entries,
                            "source": f"groq-{i+1}"
                        })
                except Exception as e:
                    err_msg = str(e)[:100]
                    errors.append(f"Groq #{i+1}: {err_msg}")
                    await ws.send_json({
                        "type": "worker", "provider": "groq", "worker_id": i + 1,
                        "status": "error", "message": f"Groq #{i+1}: {err_msg}"
                    })

        # ─── GEMINI WORKERS ───
        if gemini_keys and not all_entries:
            # Only use Gemini if Groq didn't produce results
            await ws.send_json({
                "type": "progress", "step": "gemini_upload",
                "message": f"☁️ Uploading lên Gemini File API...", "percent": 20
            })

            # Upload file once with first key
            loop = asyncio.get_event_loop()
            first_client = genai.Client(api_key=gemini_keys[0]["key"])
            uploaded_file = await loop.run_in_executor(
                None, lambda: first_client.files.upload(file=temp_path)
            )

            # Wait for processing
            for _ in range(60):
                file_info = await loop.run_in_executor(
                    None, lambda: first_client.files.get(name=uploaded_file.name)
                )
                if file_info.state.name == "ACTIVE":
                    break
                await asyncio.sleep(5)

            # Calculate total chunks = keys × threads_per_key
            n_total = len(gemini_keys) * threads_per_key
            est_duration = duration_hint if duration_hint > 0 else max(60, file_size_mb * 60)
            chunk_secs = est_duration / n_total

            time_ranges = []
            for i in range(n_total):
                start = i * chunk_secs
                end = min((i + 1) * chunk_secs + 3, est_duration + 30)  # 3s overlap
                time_ranges.append((seconds_to_mmss(start), seconds_to_mmss(end)))

            await ws.send_json({
                "type": "progress", "step": "gemini_parallel",
                "message": f"🚀 {n_total} workers ({len(gemini_keys)} keys × {threads_per_key} threads) — est. {int(est_duration)}s audio",
                "percent": 40
            })

            # Worker function — each uses its assigned key
            async def worker(idx, key_info, start_mm, end_mm):
                try:
                    client = genai.Client(api_key=key_info["key"])
                    await ws.send_json({
                        "type": "worker", "provider": "gemini", "worker_id": idx + 1,
                        "status": "running",
                        "message": f"G#{idx+1}: {start_mm}→{end_mm}"
                    })
                    entries = await gemini_transcribe_chunk(
                        client, uploaded_file, model_name, start_mm, end_mm
                    )
                    await ws.send_json({
                        "type": "worker", "provider": "gemini", "worker_id": idx + 1,
                        "status": "done",
                        "message": f"G#{idx+1}: {len(entries)} seg ✅"
                    })
                    # Stream partial results immediately
                    await ws.send_json({
                        "type": "partial",
                        "entries": entries,
                        "source": f"gemini-{idx+1}"
                    })
                    return entries
                except Exception as e:
                    err = str(e)[:100]
                    errors.append(f"G#{idx+1}: {err}")
                    await ws.send_json({
                        "type": "worker", "provider": "gemini", "worker_id": idx + 1,
                        "status": "error", "message": f"G#{idx+1}: {err}"
                    })
                    return []

            # Assign chunks to keys round-robin
            tasks = []
            for i, tr in enumerate(time_ranges):
                key_idx = i % len(gemini_keys)  # round-robin
                tasks.append(worker(i, gemini_keys[key_idx], tr[0], tr[1]))

            results = await asyncio.gather(*tasks)

            for chunk_entries in results:
                all_entries.extend(chunk_entries)

            # Cleanup uploaded file
            try:
                await loop.run_in_executor(
                    None, lambda: first_client.files.delete(name=uploaded_file.name)
                )
            except:
                pass

        elif gemini_keys and all_entries:
            # Groq already produced results, skip Gemini
            await ws.send_json({
                "type": "progress", "step": "skip_gemini",
                "message": "Groq đã xong, bỏ qua Gemini.", "percent": 80
            })

        if not all_entries:
            await ws.send_json({
                "type": "error",
                "message": f"Không tách được phụ đề. Errors: {'; '.join(errors)}"
            })
            return

        # ─── MERGE & DEDUPLICATE ───
        await ws.send_json({
            "type": "progress", "step": "merging",
            "message": f"🔄 Gộp {len(all_entries)} segments, loại bỏ trùng lặp...",
            "percent": 90
        })

        # Sort by start timestamp
        all_entries.sort(key=lambda e: ts_to_seconds(str(e.get("start", "00:00:00,000"))))

        # Remove overlapping/duplicate entries
        merged = []
        for entry in all_entries:
            if not merged:
                merged.append(entry)
                continue
            last = merged[-1]
            last_end = ts_to_seconds(str(last.get("end", "00:00:00,000")))
            curr_start = ts_to_seconds(str(entry.get("start", "00:00:00,000")))
            # If overlap > 50% of current segment, skip (duplicate)
            curr_end = ts_to_seconds(str(entry.get("end", "00:00:00,000")))
            curr_duration = curr_end - curr_start
            overlap = max(0, last_end - curr_start)
            if curr_duration > 0 and overlap / curr_duration > 0.5:
                continue  # Skip duplicate
            merged.append(entry)

        # Re-index
        for i, entry in enumerate(merged, 1):
            entry["index"] = i

        srt_content = entries_to_srt(merged)
        language = detect_language_from_entries(merged)

        await ws.send_json({
            "type": "progress", "step": "done",
            "message": f"✅ {len(merged)} phụ đề — {language} — {len(gemini_keys)}G+{len(groq_keys)}Q workers",
            "percent": 100
        })

        await ws.send_json({
            "type": "result",
            "srt": srt_content,
            "entries": merged,
            "language": language,
            "total_segments": len(merged),
            "file_name": os.path.splitext(file_name)[0] + ".srt",
            "workers_used": total_workers,
            "errors": errors,
        })

    except WebSocketDisconnect:
        logger.info(f"[WS] Session {session_id}: Client disconnected")
    except Exception as e:
        logger.error(f"[WS] Error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        try:
            await ws.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    print("SubExtractor server at http://localhost:8000")
    print("  WebSocket: ws://localhost:8000/ws/transcribe")
    uvicorn.run(
        app, host="0.0.0.0", port=8000,
        ws_max_size=500 * 1024 * 1024,  # 500 MB
        timeout_keep_alive=300,
    )
