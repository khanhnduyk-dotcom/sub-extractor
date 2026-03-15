from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import json
import os
import sys
import asyncio
import base64
import tempfile
import logging
import re
import time

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


TRANSCRIPTION_PROMPT = """You are a professional subtitle transcription engine. Transcribe ALL spoken words in this audio with precise timestamps.

CRITICAL RULES:
1. Auto-detect the spoken language. Output the transcription in the ORIGINAL spoken language (do NOT translate).
2. Each subtitle segment should be 1-2 sentences, roughly 3-8 seconds long.
3. Timestamps MUST be precise to the audio — align with actual speech start/end.
4. Include ALL spoken content, do not skip anything.
5. Output ONLY valid JSON array, no other text.

OUTPUT FORMAT (strict JSON):
[
  {"index": 1, "start": "00:00:01,200", "end": "00:00:04,800", "text": "transcribed text here"},
  {"index": 2, "start": "00:00:05,100", "end": "00:00:08,300", "text": "next segment here"}
]

TIMESTAMP FORMAT: HH:MM:SS,mmm (hours:minutes:seconds,milliseconds)
- Use commas for milliseconds separator (SRT standard)
- Be precise with timing — start when speech begins, end when it stops
- Leave natural gaps between segments where there is silence

IMPORTANT: Output ONLY the JSON array. No markdown, no explanation, no code fences."""


def parse_srt_json(text: str) -> list:
    """Parse Gemini response into list of subtitle entries."""
    # Clean response — remove markdown fences if present
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

    # Fallback: try to extract JSON array from text
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
    # Already correct format
    if re.match(r'^\d{2}:\d{2}:\d{2},\d{3}$', ts):
        return ts
    # Handle dot instead of comma
    ts = ts.replace('.', ',')
    if re.match(r'^\d{2}:\d{2}:\d{2},\d{3}$', ts):
        return ts
    # Handle MM:SS,mmm → add hours
    if re.match(r'^\d{2}:\d{2},\d{3}$', ts):
        return "00:" + ts
    # Handle MM:SS → add hours and milliseconds
    if re.match(r'^\d{2}:\d{2}$', ts):
        return "00:" + ts + ",000"
    # Handle HH:MM:SS → add milliseconds
    if re.match(r'^\d{2}:\d{2}:\d{2}$', ts):
        return ts + ",000"
    # Handle seconds only
    try:
        secs = float(ts.replace(',', '.'))
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        ms = int((secs % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    except:
        return ts


def entries_to_srt(entries: list) -> str:
    """Convert list of entries to SRT format string."""
    lines = []
    for i, entry in enumerate(entries, 1):
        idx = entry.get("index", i)
        start = format_timestamp(str(entry.get("start", "00:00:00,000")))
        end = format_timestamp(str(entry.get("end", "00:00:00,000")))
        text = entry.get("text", "").strip()
        if text:
            lines.append(f"{idx}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
    return "\n".join(lines)


def detect_language_from_entries(entries: list) -> str:
    """Simple language detection from text content."""
    if not entries:
        return "unknown"
    text = " ".join(e.get("text", "") for e in entries[:5])
    # Vietnamese
    vn_chars = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")
    if any(c in vn_chars for c in text.lower()):
        return "Tiếng Việt"
    # Chinese/Japanese/Korean
    if re.search(r'[\u4e00-\u9fff]', text):
        return "中文"
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
        return "日本語"
    if re.search(r'[\uac00-\ud7af]', text):
        return "한국어"
    # Thai
    if re.search(r'[\u0e00-\u0e7f]', text):
        return "ภาษาไทย"
    # Default to English
    return "English"


MIME_MAP = {
    '.mp3': 'audio/mp3',
    '.wav': 'audio/wav',
    '.aac': 'audio/aac',
    '.ogg': 'audio/ogg',
    '.flac': 'audio/flac',
    '.aiff': 'audio/aiff',
    '.m4a': 'audio/mp4',
    '.mp4': 'video/mp4',
    '.mkv': 'video/x-matroska',
    '.webm': 'video/webm',
    '.mov': 'video/quicktime',
}


@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    await ws.accept()
    session_id = f"ws_{int(time.time() * 1000)}"

    try:
        # 1. Receive config
        raw = await ws.receive_text()
        config = json.loads(raw)

        file_data_b64 = config.get("file_data", "")
        file_name = config.get("file_name", "audio.mp3")
        api_key = config.get("api_key", "")
        model_name = config.get("model_name", "gemini-2.5-flash")

        if not file_data_b64:
            await ws.send_json({"type": "error", "message": "No file data received"})
            return
        if not api_key:
            await ws.send_json({"type": "error", "message": "No API key provided"})
            return

        # 2. Decode and save audio file
        file_bytes = base64.b64decode(file_data_b64)
        file_size_mb = len(file_bytes) / (1024 * 1024)
        ext = os.path.splitext(file_name)[1].lower()
        mime_type = MIME_MAP.get(ext, 'audio/mp3')

        temp_path = os.path.join(TEMP_DIR, f"{session_id}_{file_name}")
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        await ws.send_json({
            "type": "progress",
            "step": "upload",
            "message": f"File received: {file_name} ({file_size_mb:.1f} MB)",
            "percent": 10
        })

        # 3. Initialize Gemini client
        client = genai.Client(api_key=api_key)

        # 4. Upload to Gemini File API
        await ws.send_json({
            "type": "progress",
            "step": "uploading",
            "message": "Uploading to Gemini...",
            "percent": 20
        })

        loop = asyncio.get_event_loop()
        uploaded_file = await loop.run_in_executor(
            None,
            lambda: client.files.upload(file=temp_path)
        )

        await ws.send_json({
            "type": "progress",
            "step": "uploaded",
            "message": f"Uploaded to Gemini ({uploaded_file.name})",
            "percent": 40
        })

        # 5. Wait for file processing
        await ws.send_json({
            "type": "progress",
            "step": "processing",
            "message": "Gemini is processing audio...",
            "percent": 50
        })

        # Poll until file is active
        for i in range(60):  # Max 5 minutes
            file_info = await loop.run_in_executor(
                None,
                lambda: client.files.get(name=uploaded_file.name)
            )
            if file_info.state.name == "ACTIVE":
                break
            await asyncio.sleep(5)

        # 6. Call Gemini for transcription
        await ws.send_json({
            "type": "progress",
            "step": "transcribing",
            "message": f"Transcribing with {model_name}...",
            "percent": 60
        })

        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model_name,
                contents=[TRANSCRIPTION_PROMPT, uploaded_file],
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 65536,
                }
            )
        )

        await ws.send_json({
            "type": "progress",
            "step": "parsing",
            "message": "Parsing transcription...",
            "percent": 85
        })

        # 7. Parse response
        response_text = response.text
        entries = parse_srt_json(response_text)

        if not entries:
            await ws.send_json({
                "type": "error",
                "message": f"Could not parse transcription. Raw response:\n{response_text[:500]}"
            })
            return

        # 8. Generate SRT
        srt_content = entries_to_srt(entries)
        language = detect_language_from_entries(entries)

        await ws.send_json({
            "type": "progress",
            "step": "done",
            "message": f"Transcription complete! {len(entries)} segments, language: {language}",
            "percent": 100
        })

        # 9. Send result
        await ws.send_json({
            "type": "result",
            "srt": srt_content,
            "entries": entries,
            "language": language,
            "total_segments": len(entries),
            "file_name": os.path.splitext(file_name)[0] + ".srt"
        })

        # Cleanup
        try:
            await loop.run_in_executor(
                None,
                lambda: client.files.delete(name=uploaded_file.name)
            )
        except:
            pass

    except WebSocketDisconnect:
        logger.info(f"[WS] Session {session_id}: Client disconnected")
    except Exception as e:
        logger.error(f"[WS] Error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        # Cleanup temp file
        try:
            if 'temp_path' in locals():
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
