# InfiniteTalk: AI Agent Codebase Guide

## Project Overview

**InfiniteTalk** is a real-time talking avatar system that streams AI-generated dialogue as continuous video. It chains three independent systems:
1. **LLM**: OpenAI/OpenRouter → text tokens (gpt-4o-mini by default)
2. **TTS**: Tokens → ElevenLabs API → MP3 audio chunks
3. **Video**: Audio frames → InfiniteTalk binary → MP4 video segments
4. **Streaming**: All segments → HLS manifest + WebSocket events → Next.js frontend

The architecture prioritizes **low-latency streaming** over batch processing—each component pushes data downstream as soon as available.

---

## Current Status (November 17, 2025)

### ✅ Working Components
- **FFmpeg idle loop**: Correctly creates HLS fMP4 segments in `src/hls/` using relative paths and finite duration (`-t 30s` with `-stream_loop 10`)
- **Backend startup**: FastAPI launches FFmpeg at startup, ensures single process, serves HLS via `/hls/`
- **WebSocket connection**: Frontend connects and receives idle segments
- **Frontend playback**: Next.js with hls.js plays idle video loop
- **Path resolution**: All paths unified to `src/hls/`, `.env` loaded from project root correctly
- **Port management**: Proper cleanup of stale uvicorn/FFmpeg processes before startup

### 🔧 Infrastructure Fixed
- FFmpeg no longer hangs with infinite looping; uses finite duration with repeated loops
- Relative paths in FFmpeg HLS output to avoid path doubling
- Single global FFmpeg process persists across WebSocket reconnections
- Environment variables correctly loaded from `.env` at project root
- Client log forwarding: hls.js errors/logs sent to backend via WebSocket

---

## Architecture: Big Picture

### Data Flow (FastAPI Backend → Next.js Frontend)

```
gpt_text_stream (LLM async generator)
    ↓ (clause-sized text chunks)
elevenlabs_stream (TTS WebSocket)
    ↓ (PCM16k s16le audio bytes)
RollingWav (audio buffering + windowing)
    ↓ (1.8s audio windows with 250ms overlap)
run_infinite_talk_window (subprocess call)
    ↓ (MP4 video segment)
_remux_mp4_to_m4s (ffmpeg remux)
    ↓ (HLS-compatible m4s segment)
HLSWriter (manifest writer)
    ↓ (WebSocket + HLS manifest)
Frontend Video.js + hls.js (streaming playback)
```

### Component Responsibilities

| File | Role | Key Pattern |
|------|------|-------------|
| `src/app/main.py` | FastAPI server + WebSocket session orchestration + global FFmpeg | Async/await; preflight checks; error handling for stubs; persistent idle FFmpeg |
| `src/app/llm_stream.py` | Token streaming with intelligent flushing | Accumulates tiny LLM tokens into clause-sized chunks (min 24 chars, punctuation triggers, lag limits) |
| `src/app/eleven_tts_ws.py` | TTS WebSocket + MP3→PCM conversion | Uses ffmpeg subprocess for zero-copy codec conversion; streaming reader task |
| `src/app/hls_writer.py` | HLS manifest writer + idle video looping | Manages sliding window of segments; discontinuity markers for idle→live transitions; FFmpeg idle segmentation |
| `src/app/controller.py` | Decoupled session controller | Dependency injection pattern (not currently used, but available for refactoring) |
| `web/src/app/page.tsx` | React frontend with HLS playback | Framer Motion animations; hls.js for Chrome; WebSocket event binding; client log forwarding |

---

## Next Steps: Feature Implementation Roadmap

### Phase 1: Live Video Generation (High Priority)
**Goal:** Replace idle loop with live AI-generated video segments.

**Tasks:**
1. **Implement prompt input** → WebSocket sends prompt to backend
   - `Session.run()` receives prompt, starts LLM stream
   - Verify `gpt_text_stream()` returns tokens correctly (or switch from stubs)
2. **Integrate TTS** → LLM tokens → audio
   - Verify `elevenlabs_stream()` works with real API key
   - Test PCM output format, sample rate, buffering
3. **Generate video segments** → audio → InfiniteTalk binary
   - Ensure `run_infinite_talk_window()` subprocess succeeds
   - Verify MP4 output from InfiniteTalk binary
4. **Remux to HLS** → ffmpeg MP4→m4s conversion
   - Test `_remux_mp4_to_m4s()` with real segments
5. **Inject live segments** → switch from idle to live in HLS manifest
   - Use `hls.force_discontinuity()` + `switch_source("live")` when first live segment ready
   - Ensure no stutter or playback interruption

**Testing Strategy:**
- Enable OPENAI_API_KEY and ELEVENLABS_API_KEY in `.env`
- Set `USE_STUBS=0` (currently stubs are enabled)
- Test prompt input flow end-to-end
- Monitor `/tmp/hls_client.log` and backend logs for errors

### Phase 2: Frontend Prompt & Controls (Medium Priority)
**Goal:** Enable user to send prompts and control session.

**Tasks:**
1. **Prompt input form** → WebSocket message
   - `onSend()` already exists in `page.tsx`; verify it sends to `/api/nop` instead of WebSocket
   - Change to send `{"type":"prompt","text":"..."}` over WebSocket
2. **Start/Stop buttons** → Session lifecycle
   - Add "Start Session" button → sends `{"type":"control","action":"start"}`
   - Add "Stop Session" button → sends `{"type":"control","action":"stop"}`
3. **Status indicator** → WebSocket events
   - Track session state (idle, generating, complete)
   - Show live/idle mode badge
4. **Error handling** → Display WebSocket and hls.js errors in UI
   - Already forwarding client logs; add visual error boundary

**Testing:**
- Test on real browser (Chrome for hls.js)
- Verify WebSocket reconnects on page refresh
- Check console for hls.js errors

### Phase 3: Audio & Timing Optimization (Medium Priority)
**Goal:** Ensure smooth audio/video sync and low latency.

**Tasks:**
1. **Audio buffering tuning** → `RollingWav` segment size
   - Current: 1.8s segments with 250ms overlap
   - Test with InfiniteTalk binary: does it expect keyframes at segment boundaries?
   - Adjust `SEG_LEN_MS` and `SEG_OVERLAP_MS` if needed
2. **LLM→TTS latency** → `gpt_text_stream()` buffering
   - Current: min 24 chars, max 450ms lag
   - Measure end-to-end latency (prompt input → audio playback)
   - Tune if >1s latency
3. **Video generation speed** → InfiniteTalk binary performance
   - Measure time from audio file → MP4 output
   - If slow, consider GPU acceleration or parallel processing
4. **HLS segment window** → `HLSWriter` sliding window
   - Current: 8 segments (default)
   - Test: are old segments deleted properly? Any stale file leaks?

**Testing:**
- Use profilers (`time`, `strace`, backend logs) to measure latency at each stage
- Monitor disk usage in `src/hls/` and `src/audio/` for cleanup

### Phase 4: Robustness & Monitoring (Lower Priority, But Important)
**Goal:** Handle edge cases and production readiness.

**Tasks:**
1. **Error recovery** → Graceful fallback to idle
   - If InfiniteTalk binary fails, continue serving idle loop
   - If TTS fails, skip segment (don't crash session)
   - Already implemented partially; test edge cases
2. **Process management** → Restart dead FFmpeg
   - Implement watchdog: detect if global FFmpeg process dies, restart it
   - Prevent zombie processes from accumulating
3. **Logging & monitoring** → Structured logs for debugging
   - Add context to all WebSocket messages (timestamp, user ID, etc.)
   - Log to file with rotation (not just stdout)
4. **Rate limiting** → Protect LLM/TTS from abuse
   - Limit prompts per minute
   - Respect API rate limits

**Testing:**
- Kill InfiniteTalk/TTS deliberately, verify fallback
- Monitor `ps aux | grep ffmpeg` for zombies
- Check `/tmp/hls_client.log` and backend logs for completeness

---

## Critical Workflows & Commands

### Local Development Setup

**Backend (Python FastAPI + InfiniteTalk binary):**
```bash
cd /home/canuk/projects/inifinitetalk-local
fuser -k 7860/tcp sleep 2 # Free port if needed
source venv/bin/activate
python -m uvicorn app.main:app --app-dir src --host 0.0.0.0 --port 7860
# Wait ~10s for FFmpeg to create idle segments
# Check: ls -lh src/hls/idle_*.m4s
```

**Frontend (Next.js):**
```bash
cd web
npm run dev
# Opens http://localhost:3001
# Connect to backend at http://localhost:7860
```

### Debugging Checklist

1. **Preflight validation** (`preflight_or_idle()` in `main.py`):
   - `which INFINITETALK_BIN` available on PATH?
   - `AVATAR_REF_IMAGE` file exists at `STATIC_DIR / ref_image_name`?
   - If either fails → session enters idle-only mode (status="idle_only")

2. **FFmpeg idle loop**:
   - Check: `ls -lh src/hls/idle_*.m4s` (should exist after ~5s)
   - Check: `curl -s http://localhost:7860/hls/idle.m3u8` (should return manifest)
   - If missing: `ps aux | grep ffmpeg` (is process alive?), check `tail -50 /tmp/backend.log`

3. **WebSocket hangs?** Check:
   - Browser console for `WS error` or `closed` events
   - Backend logs for `elevenlabs_stream` exceptions or `gpt_text_stream` failures
   - Network tab: is `/session` WebSocket connecting but receiving nothing?

4. **No video segments?** Likely `run_infinite_talk_window()` subprocess is failing:
   - Inspect stderr from `asyncio.create_subprocess_exec()` call
   - Verify audio file exists at `AUDIO_DIR/audio_####_####_####.wav`
   - Check InfiniteTalk binary version and API compatibility

5. **Video tearing/out-of-sync?** Check:
   - `SEG_LEN_MS=1800` and overlap settings match binary expectations
   - HLS discontinuity markers are correct (idle→live transition)
   - ffmpeg remux preserving keyframes: `+frag_keyframe+separate_moof`

---

## Project-Specific Patterns & Conventions

### Async/Await Throughout

**Everything is async**: LLM streams, TTS, subprocess, WebSocket sends/receives.  
Pattern: `async def` + `await` + `asyncio.gather()` / `asyncio.create_task()`

```python
# Example from Session.run():
gpt_iter = gpt_text_stream("Say hello")
tts_task = asyncio.create_task(elevenlabs_stream(gpt_iter, self.roller))
video_task = asyncio.create_task(self.video_loop())
await asyncio.gather(tts_task)  # block until TTS done
```

### Environment Configuration (`.env`)

**Critical vars** (backend expects these):
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_TEMPERATURE`
- `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL_ID`
- `INFINITETALK_BIN` (path or binary name on PATH)
- `AVATAR_REF_IMAGE` (relative to STATIC_DIR or absolute path)
- `HLS_DIR`, `AUDIO_DIR`, `STATIC_DIR` (directories created on startup)
- `SEG_LEN_MS`, `SEG_OVERLAP_MS` (timing for rolling window)
- `NEXT_PUBLIC_BACKEND_URL` (frontend only; hardcoded in env at build time)
- `USE_STUBS=0` (set to 0 for live APIs; currently 1 for testing)

**Stubs / Testing Mode**: `USE_STUBS=1` → all APIs mocked (unimplemented). Good for UI testing without API keys.

### Text Streaming Buffering Strategy

LLM tokens arrive at ~50–200ms intervals. We *cannot* send every token to TTS—TTS expects ~1–2 second phrases.

**`gpt_text_stream()` buffering logic** (`llm_stream.py`):
- Accumulates tokens into a buffer
- **Flush triggers** (any of):
  - Last token is punctuation AND buffer ≥ 8 chars
  - Buffer reaches `min_chars=24`
  - `max_lag_ms=450` exceeded (don't make user wait >0.45s)

After flush, buffer resets and cycle repeats. This ensures TTS receives natural-sounding clauses while keeping latency <0.5s.

### Audio Windowing with Overlap (RollingWav)

Audio arrives as continuous PCM32f (4 bytes/sample @ 16kHz). We **slice and overlap** to match the InfiniteTalk binary's segment model:

```
RollingWav(seg_ms=1800, overlap_ms=250):
  - Accumulate 1.8s of audio
  - Write WAV file to AUDIO_DIR
  - Keep last 250ms for next window (overlap)
  - Repeat
```

**Why overlap?** Smooth video transitions without artifacts at segment boundaries.

### HLS Manifest & Idle Looping

`HLSWriter` maintains a **sliding window** of the last N segments (default: 8). On each `add_segment()`, it rewrites `manifest.m3u8` with updated segment list.

**Idle → Live transition**:
```python
hls.switch_source("idle")  # Start pushing idle_000000.m4s, idle_000001.m4s, ...
# ... when live is ready ...
hls.force_discontinuity()   # Set pending discontinuity marker
hls.switch_source("live")   # Next add_segment() will write #EXT-X-DISCONTINUITY
```

Browser (hls.js) handles discontinuity seamlessly—no stutter.

### Error Handling: Fail Gracefully to Idle

If the InfiniteTalk binary crashes or input audio file is missing, the session **doesn't error out**. Instead:
```python
try:
    await run_infinite_talk_window(audio_slice, tmp_mp4)
except FileNotFoundError as e:
    await self.ws.send_json({"type":"status","stage":"idle_only","note":str(e)})
    # Keep idle rolling; don't crash
```

---

## Integration Points & External Dependencies

### External APIs

| Service | Endpoint | Token Limits | Notes |
|---------|----------|--------------|-------|
| OpenAI / OpenRouter | `chat.completions.create` | Rate limits vary | Fallback to `gpt-4o-mini` if not specified |
| ElevenLabs TTS | `wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input` | ~1000 requests/min | Voice ID configurable; model ID selects TTS engine |
| InfiniteTalk Binary | Local subprocess | N/A | Must be on PATH; outputs MP4 |

### Key Dependencies

**Python (`src/app/requirements.txt`)**:
- `fastapi`, `uvicorn`: Web server
- `websockets`: TTS WebSocket client
- `openai>=1.51.0`: LLM API (supports streaming)
- `python-dotenv`: `.env` loading
- `numpy`: Audio PCM conversion

**JavaScript (`web/package.json`)**:
- `next@16`: App framework
- `hls.js@1.6.14`: HLS player for Chrome
- `framer-motion`: Animations
- `tailwindcss@4`: Styling

### Subprocess Calls (Require External Tools)

1. **ffmpeg**: Audio codec conversion (MP3 → PCM) + video remuxing (MP4 → m4s) + idle loop segmentation
   ```bash
   # MP3 → PCM
   ffmpeg -f mp3 -i pipe:0 -ac 1 -ar 16000 -f s16le pipe:1
   # MP4 → m4s
   ffmpeg -y -i input.mp4 -movflags +frag_keyframe+separate_moof -f mp4 output.m4s
   # Idle loop segmentation (with finite duration to ensure output)
   ffmpeg -stream_loop 10 -i idle_loop.mp4 -t 30s -f hls -hls_segment_type fmp4 -hls_time 3 idle.m3u8
   ```

2. **InfiniteTalk binary**: Video generation from reference image + audio
   ```bash
   infinitetalk --ref_image path/to/image.png --input_audio audio.wav --output_video out.mp4 --duration_ms 1800 --no_audio
   ```

---

## Common Extensions & Modifications

### Adding a New LLM Provider

1. Modify `llm_stream.py:_make_client()` to support another `base_url` (already supports OpenRouter)
2. Update `.env` with new API key env var
3. No other changes needed—`gpt_text_stream()` is provider-agnostic

### Custom Avatar / Reference Image

1. Place image in `STATIC_DIR` (or absolute path)
2. Set `AVATAR_REF_IMAGE` env var to filename or full path
3. Restart backend; preflight checks will validate

### Adjusting Segment Timing

1. `SEG_LEN_MS=1800` (how long each video segment is)
2. `SEG_OVERLAP_MS=250` (how much overlap between segments)
3. `HLSWriter` window size in `hls_writer.py` (default: 8 segments)
4. FFmpeg idle loop duration in `hls_writer.py:start_idle_segmentation()` (currently `-t 30s`)
5. Ensure InfiniteTalk binary supports same duration; test with small value first (e.g., 500ms)

### Adding Request Logging / Telemetry

Insert logging in:
- `Session.run()`: Log session start/end, prompt input
- `elevenlabs_stream()`: Log TTS API response times, audio chunk counts
- `run_infinite_talk_window()`: Log binary exit code, stderr, duration
- `main.py`: Log WebSocket connect/disconnect, message counts

Use `logging` module; configure via `LOG_LEVEL` env var.

---

## Testing & Debugging Tips

### Manual WebSocket Test

```bash
# Terminal 1: Start backend
cd /home/canuk/projects/inifinitetalk-local
fuser -k 7860/tcp
source venv/bin/activate
python -m uvicorn app.main:app --app-dir src --host 0.0.0.0 --port 7860

# Terminal 2: Test WS with wscat
npm install -g wscat
wscat -c ws://localhost:7860/session
# You'll receive JSON events (idle segments, status)
```

### Stub Mode (Mock All APIs)

Set `USE_STUBS=1` in `.env`. Now:
- `gpt_text_stream()` → returns canned text (no OpenAI call)
- `elevenlabs_stream()` → generates noise PCM (no TTS call)
- `run_infinite_talk_window()` → creates dummy MP4 (no InfiniteTalk call)

Useful for **UI testing without API keys**.

### Check Backend Health

Visit `http://localhost:7860/` (now removed; use WebSocket test instead).  
Or check directly:
```bash
curl -s http://localhost:7860/hls/idle.m3u8
curl -s http://localhost:7860/hls/idle_000000.m4s | file -
```

---

## Code Navigation Cheatsheet

- **Main entry point**: `src/app/main.py:app` (FastAPI) + `_init_global_idle()` (FFmpeg startup)
- **WebSocket handler**: `src/app/main.py:@app.websocket("/session")`
- **Session orchestration**: `src/app/main.py:Session.run()`
- **LLM token streaming**: `src/app/llm_stream.py:gpt_text_stream()`
- **TTS streaming**: `src/app/eleven_tts_ws.py:elevenlabs_stream()`
- **Audio windowing**: `src/app/main.py:RollingWav`
- **HLS manifest**: `src/app/hls_writer.py:HLSWriter`
- **FFmpeg idle**: `src/app/hls_writer.py:HLSWriter.start_idle_segmentation()`
- **Frontend UI**: `web/src/app/page.tsx:Page()`
- **HLS playback**: `web/src/app/page.tsx:attachHls()`

---

## When Adding Features: Checklist

- [ ] Async/await used consistently?
- [ ] Environment variables documented in `.env` with sensible defaults?
- [ ] Errors logged (not silently swallowed)?
- [ ] WebSocket messages sent for UI feedback?
- [ ] Idle fallback tested (mock binary failure)?
- [ ] HLS manifest validity tested (run through `ffprobe`)?
- [ ] Frontend component mounts/unmounts cleanly?
- [ ] Rate limits respected (TTS, LLM)?
- [ ] Port conflicts handled (stale processes killed)?
- [ ] FFmpeg process lifecycle managed (single instance, graceful shutdown)?

### Data Flow (FastAPI Backend → Next.js Frontend)

```
gpt_text_stream (LLM async generator)
    ↓ (clause-sized text chunks)
elevenlabs_stream (TTS WebSocket)
    ↓ (PCM16k s16le audio bytes)
RollingWav (audio buffering + windowing)
    ↓ (1.8s audio windows with 250ms overlap)
run_infinite_talk_window (subprocess call)
    ↓ (MP4 video segment)
_remux_mp4_to_m4s (ffmpeg remux)
    ↓ (HLS-compatible m4s segment)
HLSWriter (manifest writer)
    ↓ (WebSocket + HLS manifest)
Frontend Video.js + hls.js (streaming playback)
```

### Component Responsibilities

| File | Role | Key Pattern |
|------|------|-------------|
| `src/app/main.py` | FastAPI server + WebSocket session orchestration | Async/await; preflight checks; error handling for stubs |
| `src/app/llm_stream.py` | Token streaming with intelligent flushing | Accumulates tiny LLM tokens into clause-sized chunks (min 24 chars, punctuation triggers, lag limits) |
| `src/app/eleven_tts_ws.py` | TTS WebSocket + MP3→PCM conversion | Uses ffmpeg subprocess for zero-copy codec conversion; streaming reader task |
| `src/app/hls_writer.py` | HLS manifest writer + idle video looping | Manages sliding window of segments; discontinuity markers for idle→live transitions |
| `src/app/controller.py` | Decoupled session controller | Dependency injection pattern (not currently used, but available for refactoring) |
| `web/src/app/page.tsx` | React frontend with HLS playback | Framer Motion animations; hls.js for Chrome; WebSocket event binding |

---

## Critical Workflows & Commands

### Local Development Setup

**Backend (Python FastAPI + InfiniteTalk binary):**
```bash
cd src
python -m pip install -r app/requirements.txt
# Ensure OPENAI_API_KEY, ELEVENLABS_API_KEY, INFINITETALK_BIN exported
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

**Frontend (Next.js):**
```bash
cd web
npm install
NEXT_PUBLIC_BACKEND_URL=http://localhost:7860 npm run dev
# Opens http://localhost:3000
```

### Debugging Checklist

1. **Preflight validation** (`preflight_or_idle()` in `main.py`):
   - `which INFINITETALK_BIN` available on PATH?
   - `AVATAR_REF_IMAGE` file exists at `STATIC_DIR / ref_image_name`?
   - If either fails → session enters idle-only mode (status="idle_only")

2. **WebSocket hangs?** Check:
   - Browser console for `WS error` or `closed` events
   - Backend logs for `elevenlabs_stream` exceptions or `gpt_text_stream` failures
   - Network tab: is `/session` WebSocket connecting but receiving nothing?

3. **No video segments?** Likely `run_infinite_talk_window()` subprocess is failing:
   - Inspect stderr from `asyncio.create_subprocess_exec()` call
   - Verify audio file exists at `AUDIO_DIR/audio_####_####_####.wav`

4. **Video tearing/out-of-sync?** Check:
   - `SEG_LEN_MS=1800` and overlap settings match binary expectations
   - HLS discontinuity markers are correct (idle→live transition)
   - ffmpeg remux preserving keyframes: `+frag_keyframe+separate_moof`

---

## Project-Specific Patterns & Conventions

### Async/Await Throughout

**Everything is async**: LLM streams, TTS, subprocess, WebSocket sends/receives.  
Pattern: `async def` + `await` + `asyncio.gather()` / `asyncio.create_task()`

```python
# Example from Session.run():
gpt_iter = gpt_text_stream("Say hello")
tts_task = asyncio.create_task(elevenlabs_stream(gpt_iter, self.roller))
video_task = asyncio.create_task(self.video_loop())
await asyncio.gather(tts_task)  # block until TTS done
```

### Environment Configuration (`.env`)

**Critical vars** (backend expects these):
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_TEMPERATURE`
- `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL_ID`
- `INFINITETALK_BIN` (path or binary name on PATH)
- `AVATAR_REF_IMAGE` (relative to STATIC_DIR or absolute path)
- `HLS_DIR`, `AUDIO_DIR`, `STATIC_DIR` (directories created on startup)
- `SEG_LEN_MS`, `SEG_OVERLAP_MS` (timing for rolling window)
- `NEXT_PUBLIC_BACKEND_URL` (frontend only; hardcoded in env at build time)

**Stubs / Testing Mode**: `USE_STUBS=1` → all APIs mocked (unimplemented).

### Text Streaming Buffering Strategy

LLM tokens arrive at ~50–200ms intervals. We *cannot* send every token to TTS—TTS expects ~1–2 second phrases.

**`gpt_text_stream()` buffering logic** (`llm_stream.py`):
- Accumulates tokens into a buffer
- **Flush triggers** (any of):
  - Last token is punctuation AND buffer ≥ 8 chars
  - Buffer reaches `min_chars=24`
  - `max_lag_ms=450` exceeded (don't make user wait >0.45s)

After flush, buffer resets and cycle repeats. This ensures TTS receives natural-sounding clauses while keeping latency <0.5s.

### Audio Windowing with Overlap (RollingWav)

Audio arrives as continuous PCM32f (4 bytes/sample @ 16kHz). We **slice and overlap** to match the InfiniteTalk binary's segment model:

```
RollingWav(seg_ms=1800, overlap_ms=250):
  - Accumulate 1.8s of audio
  - Write WAV file to AUDIO_DIR
  - Keep last 250ms for next window (overlap)
  - Repeat0000000000000000000000000000000000000000000000000000000000000000000000
```

**Why overlap?** Smooth video transitions without artifacts at segment boundaries.

### HLS Manifest & Idle Looping

`HLSWriter` maintains a **sliding window** of the last N segments (default: 6). On each `add_segment()`, it rewrites `manifest.m3u8` with updated segment list.

**Idle → Live transition**:
```python
hls.switch_source("idle")  # Start pushing idle_000000.m4s, idle_000001.m4s, ...
# ... when live is ready ...
hls.force_discontinuity()   # Set pending discontinuity marker
hls.switch_source("live")   # Next add_segment() will write #EXT-X-DISCONTINUITY
```

Browser (hls.js) handles discontinuity seamlessly—no stutter.

### Error Handling: Fail Gracefully to Idle

If the InfiniteTalk binary crashes or input audio file is missing, the session **doesn't error out**. Instead:
```python
try:
    await run_infinite_talk_window(audio_slice, tmp_mp4)
except FileNotFoundError as e:
    await self.ws.send_json({"type":"status","stage":"idle_only","note":str(e)})
    # Keep idle rolling; don't crash
```

---

## Integration Points & External Dependencies

### External APIs

| Service | Endpoint | Token Limits | Notes |
|---------|----------|--------------|-------|
| OpenAI / OpenRouter | `chat.completions.create` | Rate limits vary | Fallback to `gpt-4o-mini` if not specified |
| ElevenLabs TTS | `wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input` | ~1000 requests/min | Voice ID configurable; model ID selects TTS engine |
| InfiniteTalk Binary | Local subprocess | N/A | Must be on PATH; outputs MP4 |

### Key Dependencies

**Python (`src/app/requirements.txt`)**:
- `fastapi`, `uvicorn`: Web server
- `websockets`: TTS WebSocket client
- `openai>=1.51.0`: LLM API (supports streaming)
- `python-dotenv`: `.env` loading
- `numpy`: Audio PCM conversion

**JavaScript (`web/package.json`)**:
- `next@16`: App framework
- `hls.js@1.6.14`: HLS player for Chrome
- `framer-motion`: Animations
- `tailwindcss@4`: Styling

### Subprocess Calls (Require External Tools)

1. **ffmpeg**: Audio codec conversion (MP3 → PCM) + video remuxing (MP4 → m4s)
   ```bash
   ffmpeg -f mp3 -i pipe:0 -ac 1 -ar 16000 -f s16le pipe:1
   ffmpeg -y -i input.mp4 -movflags +frag_keyframe+separate_moof -f mp4 output.m4s
   ```

2. **InfiniteTalk binary**: Video generation from reference image + audio
   ```bash
   infinitetalk --ref_image path/to/image.png --input_audio audio.wav --output_video out.mp4 --duration_ms 1800 --no_audio
   ```

---

## Common Extensions & Modifications

### Adding a New LLM Provider

1. Modify `llm_stream.py:_make_client()` to support another `base_url` (already supports OpenRouter)
2. Update `.env` with new API key env var
3. No other changes needed—`gpt_text_stream()` is provider-agnostic

### Custom Avatar / Reference Image

1. Place image in `STATIC_DIR` (or absolute path)
2. Set `AVATAR_REF_IMAGE` env var to filename or full path
3. Restart backend; preflight checks will validate

### Adjusting Segment Timing

1. `SEG_LEN_MS=1800` (how long each video segment is)
2. `SEG_OVERLAP_MS=250` (how much overlap between segments)
3. Ensure InfiniteTalk binary supports same duration; test with small value first (e.g., 500ms)

### Adding Request Logging / Telemetry

Insert logging in:
- `Session.run()`: Log session start/end
- `elevenlabs_stream()`: Log TTS API response times
- `run_infinite_talk_window()`: Log binary exit code + stderr

Use `logging` module; configure via `LOG_LEVEL` env var.

---

## Testing & Debugging Tips

### Manual WebSocket Test

```bash
# Terminal 1: Start backend
cd /home/canuk/projects/inifinitetalk-local && uvicorn app.main:app --host 0.0.0.0 --port 7860
uvicorn app.main:app --app-dir src --host 0.0.0.0 --port 7860 --reload \
  --reload-exclude 'src/hls/*' --reload-exclude 'src/audio/*'


# Terminal 2: Test WS with wscat
npm install -g wscat
wscat -c ws://localhost:7860/session
# You'll receive JSON events
```

### Stub Mode (Mock All APIs)

Set `USE_STUBS=1` in `.env`. Now:
- `gpt_text_stream()` → returns canned text (no OpenAI call)
- `elevenlabs_stream()` → generates noise PCM (no TTS call)
- `run_infinite_talk_window()` → creates dummy MP4 (no InfiniteTalk call)

Useful for **UI testing without API keys**.

### Check Preflight Status

Visit `http://localhost:7860/` and check browser console for errors. Backend reports issues in the initial status message:
```json
{"type":"status","stage":"idle_only","note":"INFINITETALK_BIN not found..."}
```

---

## Code Navigation Cheatsheet

- **Main entry point**: `src/app/main.py:app` (FastAPI)
- **WebSocket handler**: `src/app/main.py:@app.websocket("/session")`
- **Session orchestration**: `src/app/main.py:Session.run()`
- **LLM token streaming**: `src/app/llm_stream.py:gpt_text_stream()`
- **TTS streaming**: `src/app/eleven_tts_ws.py:elevenlabs_stream()`
- **Audio windowing**: `src/app/main.py:RollingWav`
- **HLS manifest**: `src/app/hls_writer.py:HLSWriter`
- **Frontend UI**: `web/src/app/page.tsx:Page()`
- **HLS playback**: `web/src/app/page.tsx:attachHls()`

---

## When Adding Features: Checklist

- [ ] Async/await used consistently?
- [ ] Environment variables documented in `.env` with sensible defaults?
- [ ] Errors logged (not silently swallowed)?
- [ ] WebSocket messages sent for UI feedback?
- [ ] Idle fallback tested (mock binary failure)?
- [ ] HLS manifest validity tested (run through `ffprobe`)?
- [ ] Frontend component mounts/unmounts cleanly?
- [ ] Rate limits respected (TTS, LLM)?
