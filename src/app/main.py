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
import shutil
import uuid
import redis
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

logging.basicConfig(level=logging.DEBUG)

# Force unbuffered output
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Load .env from project root (not current directory)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)

APP_DIR  = Path(__file__).resolve().parent          # .../src/app
SRC_DIR  = APP_DIR.parent                            # .../src
ROOT_DIR = SRC_DIR.parent 
BASE_DIR = SRC_DIR
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
STATIC_DIR = Path(os.getenv("STATIC_DIR", SRC_DIR / "static"))
HLS_DIR = Path(os.getenv("HLS_DIR", "/app/src/hls"))
AUDIO_DIR  = Path(os.getenv("AUDIO_DIR",  "/app/audio"))
OUTPUT_DIR = Path("/app/output")
SEG_LEN_MS = int(os.getenv("AVATAR_SEG_MS", "1800"))
SEG_OVERLAP_MS = int(os.getenv("AVATAR_SEG_OVERLAP_MS", "250"))
INFINITETALK_BIN = os.getenv("INFINITETALK_BIN", "infinitetalk")
REF_IMAGE = os.getenv("AVATAR_REF_IMAGE", "static/me_desk.png")
IDLE_LOOP_MP4 = Path(os.getenv("IDLE_LOOP_MP4", HLS_DIR / "idle_loop.mp4"))

# Bootstrap manifest for initial client connection
BOOTSTRAP_MANIFEST = HLS_DIR / "manifest_bootstrap.m3u8"

# Redis Setup
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
AUDIO_STREAM = "audio_windows"

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    print(f"DEBUG: Connected to Redis at {REDIS_HOST}:{REDIS_PORT}", flush=True)
except Exception as e:
    print(f"ERROR: Redis connection failed: {e}", flush=True)
    redis_client = None

async def delegate_video_generation(audio_wav: Path) -> Path:
    """Delegates ML video generation to the worker via Redis."""
    if redis_client is None:
        raise RuntimeError("Redis not available for delegation")

    job_id = f"job_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    
    # Prepare message for video_worker.py
    job_msg = {
        "job_id": job_id,
        "input_data": {
            "prompt": "Live Avatar Animation",
            "cond_video": REF_IMAGE_ABS,
            "cond_audio": {
                "person1": str(audio_wav)
            }
        },
        "args": {
            "size": "infinitetalk-480",
            "motion_frame": 9,
            "frame_num": 81,
            "sample_shift": 7.0,
            "sd_steps": 8,
            "text_guide_scale": 1.0,
            "audio_guide_scale": 1.0,
            "seed": 42,
            "task": "infinitetalk-14B",
            "ulysses_size": 1,
            "ring_size": 1,
            "color_correction_strength": 1.0
        }
    }

    # Worker expects {'data': <job_as_json>}
    redis_client.xadd(AUDIO_STREAM, {"data": json.dumps(job_msg)})
    print(f"DEBUG: Dispatched job {job_id} to Redis stream {AUDIO_STREAM}", flush=True)

    # Poll for result (video_worker writes results to f"result:{job_id}")
    result_key = f"result:{job_id}"
    max_wait = 60 # worker might take time
    start = time.time()
    while time.time() - start < max_wait:
        res = redis_client.get(result_key)
        if res:
            p = Path(res)
            # Ensure path is absolute for backend to find it in the shared volume
            if not p.is_absolute():
                if str(p).startswith("output/"):
                    p = Path("/app") / p
                else:
                    p = OUTPUT_DIR / p.name
            return p
        await asyncio.sleep(0.5)

    raise TimeoutError(f"Video generation timed out for job {job_id}")



HLS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# HLS Manager: Singleton to handle manifest updates and idle synchronization for ALL sessions
class HLSManager:
    def __init__(self):
        print(f"DEBUG: HLSManager initializing. HLS_DIR={HLS_DIR}", flush=True)
        # FORCE RESET of the main manifest to avoid old live references
        with contextlib.suppress(Exception): (HLS_DIR / "manifest.m3u8").unlink()
        self.hls = HLSWriter(HLS_DIR, window=120, default_td=6.0, delete_old=True)
        self.idle_task: Optional[asyncio.Task] = None
        self.live_sessions = set() # Set of session IDs (strings)
        self.sessions = set() # Set of Session objects
        self.current_seq = 0
        self._init_done = False # Track if live_init.mp4 is already generated

        # PURGE ALL LIVE ARTIFACTS ON STARTUP to avoid stalled buffers
        print(f"DEBUG: Purging stale live files in {HLS_DIR}", flush=True)
        for p in HLS_DIR.glob("seg_*.m4s"):
            with contextlib.suppress(Exception): p.unlink()
        for p in HLS_DIR.glob("live_*.m4s"):
            with contextlib.suppress(Exception): p.unlink()
        with contextlib.suppress(Exception): (HLS_DIR / "live_init.mp4").unlink()
        self._init_done = False # Track if live_init.mp4 is already generated

    @property
    def live_active(self):
        active = len(self.live_sessions) > 0
        return active

    def register_session(self, session):
        self.sessions.add(session)
        # On first session, or if idle died, start it
        if not self.live_active:
            self.ensure_idle_loop()
            
    def unregister_session(self, session):
        self.sessions.discard(session)

    def ensure_idle_loop(self):
        # Trigger the actual FFmpeg process
        if IDLE_LOOP_MP4.exists():
            self.hls.start_idle_segmentation(IDLE_LOOP_MP4)
        
        if self.idle_task and not self.idle_task.done():
            return
        self.idle_task = asyncio.create_task(self._sync_idle_cycle())
        print("DEBUG: Master HLSManager idle loop started", flush=True)

    async def broadcast(self, msg: dict):
        # Send a message to all active WebSocket sessions
        dead = []
        for s in self.sessions:
            try:
                if s.alive:
                    await s.ws.send_json(msg)
                else:
                    dead.append(s)
            except Exception:
                dead.append(s)
        for d in dead:
            self.sessions.discard(d)

    async def _sync_idle_cycle(self):
        """Master loop that heartbeats with idle.m3u8 and updates manifest.m3u8"""
        print("DEBUG: _sync_idle_cycle TASK STARTED", flush=True)
        try:
            # Helper to parse FFmpeg's idle.m3u8
            def parse_idle_manifest():
                d_map = {}
                seq = 0
                try:
                    p = HLS_DIR / "idle.m3u8"
                    if p.exists():
                        print(f"DEBUG: Parsing {p}", flush=True)
                        lines = p.read_text(encoding="utf-8").splitlines()
                        for i, line in enumerate(lines):
                            line = line.strip()
                            if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                                try: seq = int(line.split(":")[1])
                                except: pass
                            if line.endswith(".m4s"):
                                seg_key = os.path.basename(line)
                                if i > 0 and lines[i-1].startswith("#EXTINF:"):
                                    try:
                                        dur_str = lines[i-1].split(":")[1].split(",")[0]
                                        d_map[seg_key] = float(dur_str)
                                    except: pass
                except Exception as e:
                    print(f"DEBUG: parse_idle_manifest failed: {e}", file=sys.stderr)
                return seq, d_map

            # ENSURE we start in idle mode if no live sessions
            if not self.live_active:
                self.hls.switch_source("idle")

            # Initial Sync
            _, dur_map = parse_idle_manifest()
            if dur_map:
                current_idle_segs = list(dur_map.keys())
                # The FIRST segment in our window must match the MEDIA-SEQUENCE.
                # If we show window size=20, we take the last 20.
                window_segs = current_idle_segs[-self.hls.window:]
                if window_segs:
                    # Sync media sequence to the numeric index of the first segment in our window
                    first_seg_name = window_segs[0]
                    self.hls._media_sequence = int(first_seg_name.split("_")[1].split(".")[0])
                    
                    for sname in window_segs:
                        sdur = dur_map[sname]
                        self.hls.add_segment(sname, sdur, map_file="init.mp4", delete_old=False)
                    
                    last_name = window_segs[-1]
                    self.current_seq = int(last_name.split("_")[1].split(".")[0]) + 1
            else:
                self.current_seq = 0

            # Permanent loop
            print(f"DEBUG: Entering permanent idle loop. current_seq={self.current_seq}", flush=True)
            while True:
                if self.live_active:
                    # While live is active, we just wait. 
                    # When it flips back to False, we'll hit the 'else' block below or next iteration.
                    await asyncio.sleep(1.0)
                    continue
                
                # If we just came back from live, we should force a source switch to idle
                if self.hls._current_source != "idle":
                    print("DEBUG: [HLSManager] Returning to IDLE source", flush=True)
                    self.hls.switch_source("idle", force=True)

                seg_name = f"idle_{self.current_seq:06d}.m4s"
                seg_path = HLS_DIR / seg_name
                
                if seg_path.exists():
                    print(f"DEBUG: Found idle segment {seg_name}", flush=True)
                    _, dur_map = parse_idle_manifest()
                    actual_dur = dur_map.get(seg_name, 6.0) 
                    self.hls.add_segment(seg_name, actual_dur, map_file="init.mp4", delete_old=False)
                    # Notify everyone
                    await self.broadcast({
                        "type": "video_segment",
                        "uri": f"/hls/{seg_name}",
                        "dur": actual_dur,
                        "index": self.current_seq,
                        "source": "idle"
                    })
                    print(f"DEBUG: Broadcasted {seg_name}, Next Seq: {self.current_seq + 1}", flush=True)
                    self.current_seq += 1
                else:
                    # Look for jumps
                    all_segs = sorted([f for f in os.listdir(HLS_DIR) if f.startswith("idle_") and f.endswith(".m4s")])
                    future = [int(f.split("_")[1].split(".")[0]) for f in all_segs if int(f.split("_")[1].split(".")[0]) > self.current_seq]
                    if future:
                        old_seq = self.current_seq
                        self.current_seq = min(future)
                        print(f"DEBUG: Idle sync jump: {old_seq} -> {self.current_seq}", flush=True)
                        continue
                    
                    if not all_segs:
                        # Maybe FFmpeg hasn't started writing yet?
                        pass
                    else:
                        last_exist_idx = int(all_segs[-1].split("_")[1].split(".")[0])
                        if self.current_seq > last_exist_idx + 1:
                            # We are significantly ahead, maybe FFmpeg restarted?
                            self.current_seq = last_exist_idx
                            print(f"DEBUG: Idle sync reset to tail: {self.current_seq}", flush=True)
                        elif self.current_seq < last_exist_idx:
                            # We are behind but jump logic above didn't catch it?
                            # This shouldn't normally happen if future jump works.
                            pass
                    
                    await asyncio.sleep(1.0) # Wait for FFmpeg to produce next segment

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"ERROR: HLSManager sync cycle crashed: {e}", file=sys.stderr)
            await asyncio.sleep(2.0)
            self.idle_task = None
            self.ensure_idle_loop()

# Global Singleton
hls_manager = HLSManager()

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

def preflight_check():
    problems = []

    # Ref image (make absolute)
    ref_guess = (STATIC_DIR / Path(REF_IMAGE).name) if not Path(REF_IMAGE).is_absolute() else Path(REF_IMAGE)
    if not ref_guess.exists():
        problems.append(f"AVATAR_REF_IMAGE missing: tried {_abs(ref_guess)}")

    return problems, _abs(ref_guess)

PREFLIGHT_ISSUES, REF_IMAGE_ABS = preflight_check()

app = FastAPI(title="Live Avatar Scaffold")

@app.on_event("startup")
async def startup():
    """Initialize master HLS singleton at server startup."""
    hls_manager.ensure_idle_loop()

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

@app.post("/api/test_live")
async def test_live():
    return {"ok": True}

@app.post("/debug/attach-live")
async def attach_live_test():
    """Manually trigger the Fake Live Test on the active sessions."""
    if not hls_manager.sessions:
        return {"status": "error", "message": "No active WebSocket session found. Keep frontend open."}

    triggered = 0
    for sess in list(hls_manager.sessions):
        try:
            print(f"DEBUG: Triggering fake live test on session {sess.session_id}", flush=True)
            asyncio.create_task(sess.run_fake_live_test())
            triggered += 1
        except Exception as e:
            print(f"ERROR: Failed to trigger test on session {sess.session_id}: {e}", flush=True)

    return {"status": "triggered", "sessions_triggered": triggered}



@app.websocket("/session")
async def websocket_endpoint(ws: WebSocket):
    """
    Accepts the connection, ensures the global idle FFmpeg is running,
    then creates a Session() and runs it.
    """
    await ws.accept()
    print("DEBUG: WebSocket accepted", flush=True)

    try:
        # Register session for global broadcasts
        print("DEBUG: Initializing Session object...", flush=True)
        sess = Session(ws)
        print(f"DEBUG: Session {sess.session_id} created, registering...", flush=True)
        hls_manager.register_session(sess)
        
        print(f"DEBUG: Starting sess.run() for {sess.session_id}", flush=True)
        await sess.run()
    except Exception as e:
        print(f"DEBUG: websocket_endpoint exception: {e}", flush=True)
    finally:
        if 'sess' in locals():
            print(f"DEBUG: Unregistering session {sess.session_id}", flush=True)
            hls_manager.unregister_session(sess)

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
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.roller = RollingWav(out_dir=AUDIO_DIR, seg_ms=SEG_LEN_MS, overlap_ms=SEG_OVERLAP_MS)
        self.seg_counter = 0
        self.alive = True
        self.session_id = str(uuid.uuid4())[:8]
        self.hls = hls_manager.hls
        self._live_started = False

    async def run(self):
        print(f"DEBUG: [Session {self.session_id}] run() starting", flush=True)
        recv_task = asyncio.create_task(self._recv_loop())
        video_task = asyncio.create_task(self.video_loop())
        initial_tts_task = None
        
        try:
            print(f"DEBUG: [Session {self.session_id}] Sending 'started' status...", flush=True)
            await self.ws.send_json({"type": "status", "stage": "started"})
            print(f"DEBUG: [Session {self.session_id}] Status 'started' sent", flush=True)
            
            # Push a "welcome" prompt if desired, as a background task
            try:
                print(f"DEBUG: [Session {self.session_id}] Preparing welcome TTS...", flush=True)
                gpt_iter = gpt_text_stream("Say hello")
                initial_tts_task = asyncio.create_task(elevenlabs_stream(gpt_iter, self.roller))
                print(f"DEBUG: [Session {self.session_id}] Welcome TTS task created", flush=True)
            except Exception as tts_e:
                print(f"ERROR: [Session {self.session_id}] Failed to start welcome TTS: {tts_e}", flush=True)
            
            # The session life is tied STRICTLY to the receive loop (the WS connection)
            print(f"DEBUG: [Session {self.session_id}] Awaiting recv_task...", flush=True)
            await recv_task
            print(f"DEBUG: [Session {self.session_id}] recv_task finished naturally", flush=True)
            
        except WebSocketDisconnect:
            print(f"DEBUG: WebSocket disconnected for session {self.session_id}", flush=True)
        except Exception as e:
            print(f"ERROR: Session {self.session_id} died in run(): {e}", file=sys.stderr, flush=True)
        finally:
            self.alive = False
            hls_manager.live_sessions.discard(self.session_id)
            
            print(f"DEBUG: [Session {self.session_id}] Cleaning up tasks...", flush=True)
            # Cleanup all tasks
            for name, task in [("recv", recv_task), ("video", video_task), ("tts", initial_tts_task)]:
                if task and not task.done():
                    print(f"DEBUG: [Session {self.session_id}] Cancelling {name} task", flush=True)
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            
            print(f"DEBUG: Session cleanup complete for {self.session_id}", flush=True)

    async def _remux_mp4_to_m4s(self, src_mp4: Path, dst_m4s: Path, init_map_path: Path):
        dst_m4s.parent.mkdir(parents=True, exist_ok=True)
        cwd = dst_m4s.parent
        
        # We use a robust fragmented MP4 creation command.
        # If the init file doesn't exist, we create it.
        # Otherwise, we just create the segment.
        
        flags = "frag_keyframe+empty_moov+default_base_moof"
        if not init_map_path.exists():
             # Create BOTH init and first segment
             # Using -map 0 ensures we map all streams from input.
             cmd = [
                 "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(src_mp4),
                 "-map", "0",
                 "-c", "copy", "-bsf:a", "aac_adtstoasc",
                 "-f", "mp4", "-movflags", flags,
                 str(dst_m4s)
             ]
             proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(cwd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
             stdout, stderr = await proc.communicate()
             if proc.returncode != 0:
                 raise RuntimeError(f"ffmpeg failed (init/segment): {stderr.decode()}")
             
             init_cmd = [
                 "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(src_mp4),
                 "-map", "0",
                 "-c", "copy", "-bsf:a", "aac_adtstoasc",
                 "-f", "hls", "-hls_time", "999", "-hls_segment_type", "fmp4",
                 "-hls_fmp4_init_filename", init_map_path.name,
                 "-hls_segment_filename", "tmp_init_remux_%d.m4s",
                 "tmp_init.m3u8"
             ]
             proc = await asyncio.create_subprocess_exec(*init_cmd, cwd=str(cwd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
             stdout, stderr = await proc.communicate()
             if proc.returncode != 0:
                 raise RuntimeError(f"ffmpeg failed (init cmd): {stderr.decode()}")

             # Rename the first segment to target
             if (cwd / "tmp_init_remux_0.m4s").exists():
                 (cwd / "tmp_init_remux_0.m4s").replace(dst_m4s)
             # Cleanup temp files
             with contextlib.suppress(OSError):
                 (cwd / "tmp_init.m3u8").unlink()
        else:
            # Just create the segment, ensuring it's fragmented
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(src_mp4),
                "-map", "0",
                "-c", "copy", "-bsf:a", "aac_adtstoasc",
                "-f", "mp4", "-movflags", flags,
                str(dst_m4s)
            ]
            proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(cwd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed (segment only): {stderr.decode()}")

    async def _get_media_duration(self, file_path: Path) -> float:
        try:
            # Try getting duration from format first
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, _ = await proc.communicate()
            val = stdout.decode().strip()
            if val and val != "N/A":
                return float(val)
            
            # Fallback to stream duration if format duration is N/A
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, _ = await proc.communicate()
            val = stdout.decode().strip()
            if val and val != "N/A":
                return float(val)
            
            return SEG_LEN_MS / 1000.0
        except Exception:
            return SEG_LEN_MS / 1000.0

    async def _recv_loop(self):
        print(f"DEBUG: Starting _recv_loop for session {self.session_id}", flush=True)
        while self.alive:
            try:
                data = await self.ws.receive_text()
                msg = json.loads(data)
                mtype = msg.get('type')
                if mtype != 'client_log':
                    print(f"DEBUG: Received WS message: {mtype}", flush=True)
                if mtype == 'control':
                    action = msg.get('action')
                    print(f"DEBUG: Control action: {action}", flush=True)
                    if action == 'start_idle':
                        hls_manager.ensure_idle_loop()
                    elif action == 'test_live':
                        asyncio.create_task(self.run_fake_live_test())
                elif mtype == 'prompt':
                    print(f"DEBUG: Prompt received: {msg.get('text')[:20]}...", flush=True)
                    asyncio.create_task(self._handle_prompt(msg.get('text', '')))
                elif mtype == 'client_log':
                    # Optional: log client-side errors to server stdout
                    pass
            except Exception as e:
                print(f"DEBUG: _recv_loop exception: {e}", flush=True)
                break
        print(f"DEBUG: _recv_loop finished for session {self.session_id}", flush=True)

    async def run_fake_live_test(self):
        # Use a localized flag or just allow it if we are sure we want to override
        if getattr(self, '_test_running', False):
            print("DEBUG: Fake live test already running, skipping", flush=True)
            return
        self._test_running = True
        
        print(f"DEBUG: Starting run_fake_live_test for session {self.session_id}", flush=True)
        try:
            self._live_started = True 
            hls_manager.live_sessions.add(self.session_id)
            self.hls.force_discontinuity()
            self.hls.switch_source("live_test")
            
            # Find available test segments
            ts_files = sorted([f for f in os.listdir(HLS_DIR) if f.startswith("segment_") and f.endswith(".ts")])
            if not ts_files:
                print(f"DEBUG: No segment_*.ts files found in {HLS_DIR}", flush=True)
                # Fallback to whatever segments we have
                ts_files = sorted([f for f in os.listdir(HLS_DIR) if f.endswith(".ts")])[:3]

            if not ts_files:
                print(f"ERROR: No .ts files found for live test in {HLS_DIR}", flush=True)
                return

            for i, ts_name in enumerate(ts_files):
                if not self.alive: break
                ts_path = HLS_DIR / ts_name
                m4s_name = ts_path.with_suffix('.m4s').name
                m4s_path = HLS_DIR / m4s_name
                init_name = f"live_init_{self.session_id}.mp4"
                
                print(f"DEBUG: Remuxing {ts_name} -> {m4s_name}", flush=True)
                await self._remux_mp4_to_m4s(ts_path, m4s_path, init_map_path=HLS_DIR/init_name)
                
                duration = 3.0 # Standard test duration
                self.hls.add_segment(m4s_name, duration, map_file=init_name)
                await hls_manager.broadcast({
                    "type": "video_segment",
                    "uri": f"/hls/{m4s_name}",
                    "dur": duration,
                    "index": i,
                    "source": "live_test"
                })
                await asyncio.sleep(duration)
        except Exception as e:
            print(f"ERROR: run_fake_live_test failed: {e}", flush=True)
        finally:
            self._test_running = False
            self._live_started = False
            hls_manager.live_sessions.discard(self.session_id)
            hls_manager.ensure_idle_loop()
            print(f"DEBUG: run_fake_live_test finished for session {self.session_id}", flush=True)

    async def _handle_prompt(self, prompt_text: str):
        if not prompt_text.strip(): return
        await self.ws.send_json({"type": "status", "stage": "generating", "note": prompt_text})
        gpt_iter = gpt_text_stream(prompt_text)
        await elevenlabs_stream(gpt_iter, self.roller)

    async def video_loop(self):
        try:
            while self.alive or self.roller.has_window_ready():
                if not self.roller.has_window_ready():
                    # If we WERE live, and now we are silent, release the live lock
                    # BUT NOT if a test run is active!
                    if self._live_started and not self._test_running:
                        print(f"DEBUG: [Session {self.session_id}] Silent, releasing live status", flush=True)
                        self._live_started = False
                        hls_manager.live_sessions.discard(self.session_id)
                    await asyncio.sleep(0.5); continue
                
                audio_slice = self.roller.next_window()
                print(f"DEBUG: delegating audio slice {audio_slice.name} to worker", flush=True)
                
                try:
                    # Delegate to Redis worker
                    tmp_mp4 = await delegate_video_generation(audio_slice)
                except Exception as e:
                    print(f"ERROR: delegation failed: {e}", flush=True)
                    continue
                
                seg_name = f"seg_{self.session_id}_{self.seg_counter:04d}.m4s"
                init_name = f"live_init_{self.session_id}.mp4"
                
                # PROBE SOURCE MP4 FIRST for accurate duration
                actual_dur = await self._get_media_duration(tmp_mp4)
                print(f"DEBUG: Segment {seg_name} probed duration: {actual_dur:.3f}s", flush=True)

                await self._remux_mp4_to_m4s(tmp_mp4, HLS_DIR/seg_name, init_map_path=HLS_DIR/init_name)
                
                # Delete source MP4 after remux to save space
                with contextlib.suppress(OSError): tmp_mp4.unlink()
                
                if not self._live_started:
                    self._live_started = True
                    hls_manager.live_sessions.add(self.session_id)
                    self.hls.force_discontinuity()
                    self.hls.switch_source("live")

                # Use the probed duration from original MP4, not the remuxed .m4s which ffprobe often misreads
                self.hls.add_segment(seg_name, actual_dur, map_file=init_name)
                await hls_manager.broadcast({
                    "type": "video_segment",
                    "uri": f"/hls/{seg_name}",
                    "index": self.seg_counter,
                    "dur": actual_dur,
                    "source": "live"
                })
                self.seg_counter += 1
                await asyncio.sleep(actual_dur)
        finally:
            hls_manager.live_sessions.discard(self.session_id)
            print(f"DEBUG: video_loop finished for session {self.session_id}", flush=True)

# Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
