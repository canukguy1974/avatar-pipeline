# /app/main.py
# Fixed invalid f-string with embedded JS. Using triple quotes for literal HTML/JS block to avoid SyntaxError.
import mimetypes
mimetypes.add_type("application/vnd.apple.mpegurl", ".m3u8")
mimetypes.add_type("video/mp4", ".m4s")
import asyncio, base64
import contextlib
from typing import List, Tuple
import json, logging
import websockets
import os
import time
from pathlib import Path
import wave, io, struct
import subprocess
from shutil import which
from app.hls_writer import HLSWriter
from app.controller import run_controller
from app.eleven_tts_ws import elevenlabs_stream
from typing import AsyncIterator, Optional
from app.llm_stream import gpt_text_stream

from websockets.exceptions import InvalidStatusCode, ConnectionClosed
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from dotenv import load_dotenv

# Load .env from project root (not current directory)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)

APP_DIR  = Path(__file__).resolve().parent          # .../src/app
SRC_DIR  = APP_DIR.parent                            # .../src
ROOT_DIR = SRC_DIR.parent 
BASE_DIR = SRC_DIR
USE_STUBS = os.getenv("USE_STUBS", "1") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
STATIC_DIR = Path(os.getenv("STATIC_DIR", SRC_DIR / "static"))
HLS_DIR    = Path(os.getenv("HLS_DIR",    "/home/canuk/projects/inifinitetalk-local/src/hls"))
AUDIO_DIR  = Path(os.getenv("AUDIO_DIR",  "/home/canuk/projects/inifinitetalk-local/src/audio"))
SEG_LEN_MS = int(os.getenv("AVATAR_SEG_MS", "1800"))
SEG_OVERLAP_MS = int(os.getenv("AVATAR_SEG_OVERLAP_MS", "250"))
INFINITETALK_BIN = os.getenv("INFINITETALK_BIN", "infinitetalk")
REF_IMAGE = os.getenv("AVATAR_REF_IMAGE", "static/me_desk.png")
IDLE_LOOP_MP4 = Path(os.getenv("IDLE_LOOP_MP4", HLS_DIR / "idle_loop.mp4"))

# Bootstrap manifest for initial client connection
BOOTSTRAP_MANIFEST = HLS_DIR / "manifest_bootstrap.m3u8"



HLS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Global idle segmentation process (persistent across WebSocket sessions)
_global_idle_proc = None
_global_hls_writer = None

def _init_global_idle():
    """Initialize the global idle FFmpeg process at startup (called once)."""
    global _global_idle_proc, _global_hls_writer
    print(f"DEBUG: HLS_DIR = {HLS_DIR}", flush=True)
    print(f"DEBUG: IDLE_LOOP_MP4 = {IDLE_LOOP_MP4}", flush=True)
    print(f"DEBUG: IDLE_LOOP_MP4.exists() = {IDLE_LOOP_MP4.exists()}", flush=True)
    
    if _global_idle_proc is not None:
        return  # Already running
    
    print(f"DEBUG: Initializing global idle FFmpeg", flush=True)
    _global_hls_writer = HLSWriter(HLS_DIR, window=8, default_td=3.0, delete_old=True)
    
    if IDLE_LOOP_MP4.exists():
        try:
            _global_hls_writer.start_idle_segmentation(IDLE_LOOP_MP4)
            _global_idle_proc = _global_hls_writer._idle_proc
            print(f"DEBUG: Global idle FFmpeg started (PID: {_global_idle_proc.pid})", flush=True)
        except Exception as e:
            print(f"DEBUG: Failed to start global idle FFmpeg: {e}", flush=True)
    else:
        print(f"DEBUG: Idle loop file not found: {IDLE_LOOP_MP4}", flush=True)

def _create_bootstrap_manifest():
    """Create a minimal HLS manifest for initial client connection."""
    content = """#EXTM3U
#EXT-X-VERSION:7
#EXT-X-TARGETDURATION:2
#EXT-X-MEDIA-SEQUENCE:0
"""
    BOOTSTRAP_MANIFEST.write_text(content, encoding="utf-8")

_create_bootstrap_manifest()

def _abs(p: Path | str) -> str:
    return str((Path(p).resolve() if isinstance(p, (str, Path)) else p))

def preflight_or_idle():
    problems = []

    # Binary
    from shutil import which
    if which(INFINITETALK_BIN) is None:
        problems.append(f"INFINITETALK_BIN not found on PATH: {INFINITETALK_BIN}")

    # Ref image (make absolute)
    ref_guess = (STATIC_DIR / Path(REF_IMAGE).name) if not Path(REF_IMAGE).is_absolute() else Path(REF_IMAGE)
    if not ref_guess.exists():
        problems.append(f"AVATAR_REF_IMAGE missing: tried {_abs(ref_guess)}")

    return problems, _abs(ref_guess)

PREFLIGHT_ISSUES, REF_IMAGE_ABS = preflight_or_idle()

app = FastAPI(title="Live Avatar Scaffold")

@app.on_event("startup")
async def startup():
    """Initialize global idle FFmpeg at server startup."""
    _init_global_idle()

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.api_route("/hls/{filename}", methods=["GET", "HEAD"])
async def serve_hls(filename: str):
    path = HLS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404)

    if filename.endswith(".m3u8"):
        return FileResponse(
            path,
            media_type="application/vnd.apple.mpegurl",
            headers={
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*",
            },
        )

    if filename.endswith(".ts"):
        return FileResponse(
            path,
            media_type="video/mp2t",
            headers={
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*",
            },
        )

    return FileResponse(path)


@app.post("/api/nop")
async def nop():
    return {"ok": True}


@app.websocket("/session")
async def session(ws: WebSocket):
    """WebSocket endpoint for frontend sessions.
    Accepts the connection, ensures the global idle FFmpeg is running,
    then creates a Session() and runs it.
    """
    await ws.accept()
    print("DEBUG: WebSocket accepted", flush=True)

    # Ensure global idle process is running
    _init_global_idle()

    # Use the global HLS writer if available
    hls = _global_hls_writer if _global_hls_writer else HLSWriter(HLS_DIR, window=8, default_td=3.0, delete_old=True)
    print(f"DEBUG: Using HLS writer (default_td={hls.default_td})", flush=True)

    # Wait briefly for manifest/init files to appear
    manifest_path = HLS_DIR / "idle.m3u8"
    init_path = HLS_DIR / "init.mp4"
    max_wait = 50
    for attempt in range(max_wait):
        if manifest_path.exists() and init_path.exists():
            print(f"DEBUG: Manifest and init ready after {(attempt+1)*100}ms", flush=True)
            break
        await asyncio.sleep(0.1)

    if not manifest_path.exists():
        print("DEBUG: Manifest still not ready after wait", flush=True)

    await ws.send_json({"type": "status", "stage": "idle_setup", "note": "Idle loop ready"})

    sess = Session(ws, hls)
    print("DEBUG: Creating Session", flush=True)
    try:
        await sess.run()
    except Exception as e:
        print(f"DEBUG: session run error: {e}", flush=True)

class RollingWav:
    def __init__(self, out_dir: Path, seg_ms: int, overlap_ms: int, sr=16000):
        self.out_dir = out_dir; self.seg_ms = seg_ms; self.overlap_ms = overlap_ms
        self.sr = sr; self.seg_index = 0; self._pcm = bytearray()  # f32le mono

    def write_pcm16k_s16le(self, pcm_bytes: bytes):
        import numpy as np

        f32 = (
            np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        ).tobytes()
        self._pcm.extend(f32)

    async def write_pcm_16k_f32le(self, pcm_bytes: bytes):
        self._pcm.extend(pcm_bytes)

    def has_window_ready(self) -> bool:
        want = int(self.seg_ms * self.sr / 1000)
        have = len(self._pcm) // 4
        return have >= want - int(self.overlap_ms * self.sr / 1000)

    def next_window(self) -> Optional[Path]:
        if not self.has_window_ready(): return None
        hop = self.seg_ms - self.overlap_ms
        n_total = int(self.seg_ms * self.sr / 1000)
        n_hop = int(hop * self.sr / 1000)
        # slice last seg_len samples
        window = self._pcm[: n_total*4]
        # keep overlap for next window
        self._pcm = self._pcm[n_hop*4 :]

        start_ms = self.seg_index * hop
        end_ms = start_ms + self.seg_ms
        p = self.out_dir / f"audio_{self.seg_index:04d}_{start_ms}_{end_ms}.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self.sr)
            # convert f32 -> int16 for compatibility
            import numpy as np
            f = np.frombuffer(window, dtype=np.float32)
            i16 = (np.clip(f, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            wf.writeframes(i16)
        self.seg_index += 1
        return p

        
class Session:
    def __init__(self, ws: WebSocket, hls: HLSWriter):
        self.ws = ws
        self.hls = hls
        self.roller = RollingWav(AUDIO_DIR, SEG_LEN_MS, SEG_OVERLAP_MS)
        self.seg_counter = 0
        self.alive = True
        self._live_started = False

    async def _seed_idle_until_live(self):
        """Read the files created by start_idle_segmentation() and feed them until live begins."""
        print(f"DEBUG: _seed_idle_until_live started", flush=True)
        self.hls.switch_source("idle")
        idle_segment_duration = 3.0  # Match FFmpeg -hls_time setting (approx)
        
        # Find any available segments
        import os
        def get_segments():
            return sorted([f for f in os.listdir(HLS_DIR) if f.startswith("idle_") and f.endswith(".m4s")])

        existing_segments = get_segments()
        print(f"DEBUG: Found {len(existing_segments)} idle segments", flush=True)
        
        # Wait a bit if none found initially
        if not existing_segments:
            await asyncio.sleep(1.0)
            existing_segments = get_segments()
            print(f"DEBUG: After wait, found {len(existing_segments)} idle segments", flush=True)
        
        if not existing_segments:
            print(f"DEBUG: No idle segments available, sending status", flush=True)
            await self.ws.send_json({"type": "status", "note": "No idle segments available"})
            return
        
        # Start from a recent segment to avoid playing ancient history or getting stuck in gaps too far back.
        # If we have many segments, start from the last 5 or so.
        start_index = 0
        if len(existing_segments) > 6:
            start_index = len(existing_segments) - 6
        
        first_seg_name = existing_segments[start_index]
        current_seq = int(first_seg_name.split("_")[1].split(".")[0])
        print(f"DEBUG: Starting from segment sequence {current_seq} ({first_seg_name})", flush=True)
        
        # Continuous streaming loop
        while not self._live_started and self.alive:
            seg_name = f"idle_{current_seq:06d}.m4s"
            seg_path = HLS_DIR / seg_name
            
            if seg_path.exists():
                print(f"DEBUG: Adding segment {seg_name}", flush=True)
                # Try to get actual duration if possible, or use default
                # (For now fixed default is fine as they are generated uniformly)
                self.hls.add_segment(seg_name, idle_segment_duration)
                await self.ws.send_json({"type":"video_segment","uri":f"/hls/{seg_name}","dur":idle_segment_duration})
                current_seq += 1
            else:
                # Segment missing. Check if there are NEWER segments (gap jump).
                # Re-scan directory to see if we fell behind or hit a gap.
                all_segs = get_segments()
                # Filter for segments with sequence > current_seq
                future_segs = []
                for f in all_segs:
                    try:
                        s_seq = int(f.split("_")[1].split(".")[0])
                        if s_seq > current_seq:
                            future_segs.append(s_seq)
                    except (ValueError, IndexError):
                        pass
                
                if future_segs:
                    next_seq = min(future_segs)
                    print(f"DEBUG: Gap detected. Current {current_seq} missing, jumping to {next_seq}", flush=True)
                    current_seq = next_seq
                    # Immediate continue to process this new segment
                    continue
                else:
                    # No future segments found, we are likely at the live edge waiting for ffmpeg
                    await asyncio.sleep(0.1)

        print(f"DEBUG: _seed_idle_until_live exited", flush=True)

    async def run(self):
        # Start a receiver task so client can send us logs (client_log messages)
        recv_task = asyncio.create_task(self._recv_loop())
        try:
            await self.ws.send_json({"type": "status", "stage": "started"})
            gpt_iter = gpt_text_stream("Say hello")
            tts_task = asyncio.create_task(elevenlabs_stream(gpt_iter, self.roller))
            video_task = asyncio.create_task(self.video_loop())
            # Wait for TTS to finish; video loop will keep running until alive==False
            await asyncio.gather(tts_task)
            await asyncio.sleep(1.0)
            self.alive = False
            await video_task
            await self.ws.send_json({"type": "done"})
        except WebSocketDisconnect:
            self.alive = False
        except Exception as e:
            try:
                await self.ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
            self.alive = False
        finally:
            recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recv_task

    async def _remux_mp4_to_m4s(self, src_mp4: Path, dst_m4s: Path):
        dst_m4s.parent.mkdir(parents=True, exist_ok=True)
        # Run ffmpeg asynchronously to avoid blocking the asyncio event loop
        # Use movflags that are compatible with Chrome MSE / hls.js for fragmented MP4:
        # +default_base_moof ensures correct base offsets for moof boxes, combine with frag_keyframe and separate_moof
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src_mp4),
            "-map", "0:v:0", "-c:v", "copy", "-an",
            "-movflags", "+default_base_moof+frag_keyframe+separate_moof",
            "-f", "mp4", str(dst_m4s)
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg remux failed (rc={proc.returncode}): {err.decode('utf-8', errors='replace')}")

    async def _recv_loop(self):
        """Receive messages from client (e.g. forwarded hls.js logs) and persist them for inspection."""
        log_path = Path('/tmp/hls_client.log')
        while self.alive:
            try:
                data = await self.ws.receive_text()
            except Exception:
                break
            try:
                msg = json.loads(data)
            except Exception:
                # write raw line
                try:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open('a', encoding='utf-8') as f:
                        f.write(f"{time.time()}: RAW: {data}\n")
                except Exception:
                    pass
                continue

            if isinstance(msg, dict):
                mtype = msg.get('type')
                if mtype == 'client_log':
                    level = msg.get('level', 'info')
                    text = msg.get('msg') or msg.get('message') or str(msg)
                    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {text}\n"
                    try:
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with log_path.open('a', encoding='utf-8') as f:
                            f.write(line)
                        print(f"CLIENT_LOG: {line.strip()}", flush=True)
                    except Exception:
                        print(f"CLIENT_LOG_WRITE_FAILED: {line.strip()}", flush=True)
                elif mtype == 'control':
                    action = msg.get('action')
                    try:
                        if action == 'start_idle':
                            if IDLE_LOOP_MP4.exists():
                                self.hls.start_idle_segmentation(IDLE_LOOP_MP4)
                                await self.ws.send_json({"type": "status", "stage": "idle_running", "note": "idle started"})
                                print("DEBUG: idle started via control", flush=True)
                            else:
                                await self.ws.send_json({"type": "status", "stage": "idle_missing", "note": f"Idle loop file not found: {IDLE_LOOP_MP4}"})
                        elif action == 'stop_idle':
                            self.hls.stop_idle_segmentation()
                            await self.ws.send_json({"type": "status", "stage": "idle_stopped", "note": "idle stopped"})
                            print("DEBUG: idle stopped via control", flush=True)
                    except Exception as e:
                        print(f"DEBUG: error handling control msg: {e}", flush=True)
                elif mtype == 'prompt':
                    prompt_text = msg.get('text', '')
                    if prompt_text.strip():
                        print(f"DEBUG: Received prompt: {prompt_text}", flush=True)
                        # Handle prompt asynchronously so we don't block recv loop
                        asyncio.create_task(self._handle_prompt(prompt_text))
                        await self.ws.send_json({"type": "status", "stage": "accepted", "note": "Prompt accepted"})
                    else:
                        await self.ws.send_json({"type": "error", "message": "Prompt is empty"})
                else:
                    print(f"WS RECV: {msg}", flush=True)

    async def _handle_prompt(self, prompt_text: str):
        """Run the LLM -> TTS -> Video generation pipeline for a single prompt."""
        try:
            await self.ws.send_json({"type": "status", "stage": "generating", "note": "Starting prompt processing"})
            gpt_iter = gpt_text_stream(prompt_text)
            tts_task = asyncio.create_task(elevenlabs_stream(gpt_iter, self.roller))
            video_task = asyncio.create_task(self.video_loop())
            await asyncio.gather(tts_task)
            await asyncio.sleep(1.0)
            self.alive = False
            await video_task
            await self.ws.send_json({"type": "done"})
        except Exception as e:
            print(f"ERROR: Prompt processing failed: {e}", flush=True)
            try:
                await self.ws.send_json({"type": "error", "message": f"Prompt processing failed: {str(e)}"})
            except Exception:
                pass

    async def video_loop(self):
        idle_task = asyncio.create_task(self._seed_idle_until_live())
        first_live_done = False
        try:
            while self.alive or self.roller.has_window_ready():
                if not self.roller.has_window_ready():
                    await asyncio.sleep(0.05)
                    continue

                audio_slice = self.roller.next_window()
                if not audio_slice:
                    continue

                tmp_mp4 = HLS_DIR / f"chunk_{self.seg_counter:04d}.mp4"
                await run_infinite_talk_window(audio_slice, tmp_mp4)

                seg_name = f"seg_{self.seg_counter:04d}.m4s"
                seg_path = HLS_DIR / seg_name
                await self._remux_mp4_to_m4s(tmp_mp4, seg_path)
                with contextlib.suppress(Exception):
                    tmp_mp4.unlink()

                if not first_live_done:
                    self._live_started = True
                    self.hls.force_discontinuity()
                    self.hls.switch_source("live")
                    first_live_done = True

                self.hls.add_segment(seg_name, SEG_LEN_MS / 1000.0)
                await self.ws.send_json({"type":"video_segment","uri":f"/hls/{seg_name}","index":self.seg_counter,"dur_ms":SEG_LEN_MS})
                self.seg_counter += 1
                await asyncio.sleep(SEG_LEN_MS / 1000.0)
        except FileNotFoundError as e:
            if not first_live_done:
                await self.ws.send_json({"type":"status","stage":"idle_only","note":str(e)})
        finally:
            idle_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await idle_task
async def run_infinite_talk_window(audio_wav: Path, out_mp4: Path):
    # If preflight failed, bail early so we stay in idle and keep UI clean
    if PREFLIGHT_ISSUES:
        raise FileNotFoundError("; ".join(PREFLIGHT_ISSUES))

    cmd = [
        INFINITETALK_BIN,
        "--ref_image", REF_IMAGE_ABS,
        "--input_audio", str(audio_wav),
        "--output_video", str(out_mp4),
        "--duration_ms", str(SEG_LEN_MS),
        "--no_audio",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"InfiniteTalk failed: {stderr.decode()}")   

# Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
