# InfiniteTalk — Project Architecture Blueprint

Generated: 2025-11-18

Purpose: provide a clear, implementation-ready architecture for how I would build InfiniteTalk from scratch, evaluate the current approach, and give a step-by-step plan to get to a robust, maintainable, production-ready system.

Summary recommendation
- The overall design is appropriate: an async streaming pipeline (LLM → TTS → audio windows → video generator → HLS) is the correct pattern for low-latency talking-avatar streaming. The repository already follows this flow and uses appropriate tools (FastAPI, ffmpeg, Next.js).
- To harden and scale the product, split responsibilities into well-defined components (ingest / control, LLM streaming, TTS worker, audio windowing, video generation worker, HLS writer, and frontend). Use a lightweight message bus for decoupling (Redis streams or RabbitMQ) and containerize components.
- Priorities: 1) Stabilize process lifecycle (global FFmpeg), 2) decouple long-running/CPU-bound jobs into workers, 3) add observability and graceful error handling, 4) add CI/CD and infra for deployments.

Table of Contents
1. Architecture Overview
2. Component Breakdown
3. Data Flow (detailed)
4. Implementation Patterns & Interfaces
5. Cross-cutting Concerns
6. Deployment & Infrastructure
7. Testing Strategy
8. Step-by-step Implementation Plan (from scratch)
9. Decision Records & Rationale
10. Extensions and Evolution

---

1. Architecture Overview

- Pattern: Event-driven, modular monolith → microservice migration path. Start with clear process boundaries (separate worker processes) while keeping a single repo and Docker Compose for development. Move to Kubernetes when scaling.
- Major subsystems:
  - Control plane / API (FastAPI) — manages sessions, WebSocket control channel, user input, and statuses.
  - LLM text stream (service or client lib) — produces clause-sized text events.
  - TTS worker (ElevenLabs) — converts clause text → audio bytes/streams.
  - Audio windowing (RollingWav) — accumulates audio into overlapping windows and writes WAV slices.
  - Video generator worker (InfiniteTalk binary) — consumes audio window and produces MP4 segments.
  - HLS remuxer/writer (ffmpeg + HLSWriter) — remuxes MP4 → fMP4 segments + manifests and serves via HTTP or object store.
  - Frontend (Next.js + hls.js) — playback, controls, and logs.
  - Message bus (Redis Streams or RabbitMQ) — decouple components and provide backpressure and persistence.

Why this layout
- Low latency needs pipelining: short text flushes should not block video generation.
- Offloading CPU-bound video creation to workers prevents blocking the API event loop.
- A message bus gives resiliency, easier retries, and observability.


2. Component Breakdown

- API / Control (FastAPI)
  - Responsibilities: WebSocket acceptor (`/session`), session lifecycle, authentication (if needed), and orchestrating the session (publishing messages to the bus and controlling HLSWriter).
  - Interfaces: REST endpoints for health, metrics, and optional HTTP fallback for prompts; WebSocket messages for `type: prompt`, `type: control`, `type: client_log`.
  - Important: Keep WebSocket handler thin — only accept/validate messages and publish to bus. Avoid running long CPU work in the handler.

- LLM Streamer
  - Responsibilities: receive prompt, stream tokens from chosen LLM provider, and flush clause-sized strings into a topic `llm_out`.
  - Implementation: either integrated library inside API (small scale) or a dedicated worker if you need independent scaling. Use an async client with retry/backoff.

- TTS Worker
  - Responsibilities: subscribe to `llm_out`, call ElevenLabs (or other TTS) streaming API, perform MP3 → PCM conversion (ffmpeg subprocess or native libs), and publish PCM audio frames to `audio_stream` topics.
  - Implementation detail: prefer a worker that writes PCM to an in-memory ring buffer or directly to `RollingWav` service.

- RollingWav (Audio Windowing)
  - Responsibilities: accumulate PCM audio into overlapping windows (e.g., 1800ms with 250ms overlap), write WAV files or provide in-memory buffers for downstream worker.
  - Implementation: as a small local library used by TTS worker or separate process; ensure consistent sample rates and channels (16kHz mono s16le).

- Video Generator Worker
  - Responsibilities: consume WAV windows, call InfiniteTalk binary (or containerized GPU process) to generate MP4; validate outputs; publish segment artifacts to `hls_input`.
  - Implementation: put this worker behind a queue (e.g., Redis queue). It must be robust to failures and support retries.

- HLS Writer
  - Responsibilities: remux MP4 → fMP4 segments with ffmpeg (use `-movflags +frag_keyframe+separate_moof`), write `init.mp4`, `segment.m4s`, update manifest, rotate, and optionally upload to object storage.
  - Implementation: run in a dedicated process with careful cwd and relative filenames; use atomic manifest writes.

- Global Idle FFmpeg
  - Keep the idle loop as an independent ffmpeg process invoked with a working directory pointing to HLS dir; do not rely on API event loop to manage it. Monitor and restart if it dies. Prefer systemd/unit or process manager in production (or container restart policy).

- Frontend (Next.js + hls.js)
  - Responsibilities: connect via WebSocket to API, request prompts, attach hls.js to video element, forward client logs to backend.
  - Implementation: keep playback decoupled — let HLS writer determine URLs. Use `NEXT_PUBLIC_BACKEND_URL`.


3. Data Flow (detailed)

- User types prompt → UI sends WebSocket JSON `{type: 'prompt', text}` → API validates and publishes `session:{id}:prompt` to bus.
- LLM Streamer reads `session:{id}:prompt`, streams tokens, accumulates clause-sized text and publishes `session:{id}:clause` messages (fields: clause_id, text, timestamp).
- TTS Worker subscribes to `session:{id}:clause`, requests ElevenLabs streaming endpoint, receives MP3 chunks, converts to PCM via ffmpeg subprocess (pipe), and writes PCM to audio buffer.
- RollingWav reads PCM buffer and every SEG_LEN_MS creates a WAV window file: `audio_{session}_{seq}.wav` and publishes `audio_window` event with path/metadata.
- Video Generator Worker picks up `audio_window` events, runs InfiniteTalk binary to create `out_{session}_{seq}.mp4` and publishes `mp4_ready` event.
- HLS Writer listens for `mp4_ready`, remuxes to `init.mp4` and `seg_{seq}.m4s`, updates `manifest.m3u8`, and notifies API via `segment_published` event; API forwards segment updates over WS to browser (optionally) so UI can switch from idle to live.

Key constraints
- Maintain sample rate and audio format invariants across stages.
- Keep segments aligned with HLS times (`hls_time`) and ensure keyframes are preserved.
- Ensure overlap windows are handled consistently so video generator does not produce choppy motion.


4. Implementation Patterns & Interfaces

- Messaging topics (Redis Streams example):
  - `session:{id}:prompts` — prompt commands
  - `session:{id}:llm_clauses` — flushed clause text events
  - `session:{id}:audio_windows` — WAV file metadata for downstream
  - `session:{id}:mp4_ready` — mp4 path and metadata
  - `hls:segments` — published segments for manifest updates

- Worker contract examples (pseudo):
  - LLM => publishes: {clause_id, text, chars, flush_reason, time}
  - TTS => publishes: {wav_path, seq, duration_ms}
  - VideoWorker => publishes: {mp4_path, seq, duration_ms}
  - HLSWriter => publishes: {segment_name, init_name, manifest_version}

- Idempotency & dedupe: each worker should use sequence numbers and unique filenames (session + seq) so retries are safe.

- Graceful termination: workers must finish current item and flush metrics on SIGTERM.


5. Cross-cutting Concerns

- Secrets & config
  - Use `python-dotenv` for local dev; use environment variables in containers and a secret manager in prod (Vault, AWS Secrets Manager).

- Observability
  - Structured logging (JSON), correlate with `session_id` and `trace_id`.
  - Metrics: expose Prometheus metrics from API and workers (request latency, queue depth, segment publish rate, FFmpeg exit codes).
  - Tracing: use OpenTelemetry to trace from prompt → LLM → TTS → HLS segment.

- Error handling
  - Retries with exponential backoff for external APIs (LLM, TTS). Circuit-breaker for repeated failures.
  - If video generation fails, fallback to idle segment and surface status via WS.

- Security
  - Sanitize incoming prompts if they get executed anywhere. Require HTTPS in production. Rate-limit prompt submissions per session/IP.
  - Avoid publicly exposing raw ffmpeg/InfiniteTalk interfaces.


6. Deployment & Infrastructure

- Local dev: Docker Compose with services: `api`, `llm_worker` (optional), `tts_worker`, `video_worker`, `hls_writer`, `redis`, `postgres` (if needed for metadata), `nginx` reverse proxy for HLS static hosting.
- Prod: Kubernetes with two tiers:
  - Control plane: `api` (fastapi) behind ingress (TLS), autoscaled horizontally.
  - Worker pool: `tts_worker`, `video_worker` as deployments with HPA. Use GPU node pool for video generator if InfiniteTalk can use GPU.
  - Storage: S3-compatible object store for HLS segments (optionally) with a CDN in front.
  - Message bus: Redis managed or RabbitMQ cluster.
- Process management: run ffmpeg/idle loop in its own container with restart policy.
- **Model Caching (RunPod)**: Use RunPod's cached models feature to reduce cold starts. Models are stored at `/runpod-volume/huggingface-cache/hub/`. Use `src/app/runpod_utils.py` to programmatically locate these models in workers.


7. Testing Strategy

- Unit tests: isolate small modules (RollingWav logic, message serialization, manifest writer). Use `pytest` for Python.
- Integration tests: Docker Compose-based E2E tests that simulate prompt → playback using headless browser tests (Playwright) to validate HLS playback.
- Contract tests: define and test message schemas between components (e.g., JSON schema for `llm_clause`).
- Load tests: k6 or Locust to measure prompt ingestion and worker scaling.


8. Step-by-step Implementation Plan (from scratch)

Phase A — Minimal Viable Pipeline (MVP)
1. Scaffold monorepo structure:
   - `api/` (FastAPI), `workers/` (Python modules), `web/` (Next.js), `infra/` (docker-compose)
2. Implement FastAPI WebSocket control handler and a simple idle HLS writer that runs ffmpeg with a fixed loop.
3. Implement stub LLM and TTS (stubs return canned text/audio) and RollingWav to write windows and verify video generator can accept them.
4. Use a simple local queue (asyncio.Queue or Redis) to pass messages between in-process components.
5. Implement HLSWriter to produce valid `manifest.m3u8` + `init.mp4` + `seg.m4s` and attach in frontend via hls.js.
6. Validate front-to-back E2E in local Docker Compose.

Phase B — Workerization & Robustness
1. Extract TTS and Video generator into separate processes/workers consuming Redis streams.
2. Add durable queues and a small metadata DB (sqlite/postgres) to track session state.
3. Add observability (Prometheus metrics endpoint) and structured logs.
4. Replace stubs with real LLM & ElevenLabs integration and tune LLM flush rules.
5. Add retry/circuit breaker logic for external API calls.

Phase C — Productionization
1. Containerize components and create Helm charts or Kubernetes manifests.
2. Add autoscaling policies and GPU node pool for video generation.
3. Move HLS segments to S3 with CDN and ensure manifest and segment naming supports partial updates.
4. Harden security: TLS, authentication, rate limiting, secrets manager.
5. Create CI pipelines for lint/test/build and CD for deployments.

Phase D — Scale & UX
1. Advanced scheduling and prewarming of video generators (cache reference embeddings, warm models) to lower latency.
2. Add session replay, user analytics, and monitoring dashboards.
3. Implement multi-tenant isolation and billing hooks if needed.


9. Decision Records & Rationale (high level)

- Use event-driven / message bus: decouples producers/consumers, enables retries, backpressure and independent scaling. Alternative (direct synchronous calls) would cause the API to be fragile under heavy CPU/video load.
- Use separate process for ffmpeg idle loop: simplifies lifecycle and makes ffmpeg independent of Python event loop. Alternative (spawn per session) would create process churn and complexity.
- Use Redis Streams vs RabbitMQ: Redis is simpler to operate and integrates well with Python async; RabbitMQ offers richer routing and ack semantics. Choose Redis unless you need advanced routing guarantees.
- Use S3+CDN for segments when scaling: avoids single-server I/O bottlenecks and lets CDN accelerate playback globally.


10. Extensions & Evolution

- Support multiple LLM/TTS providers via an adapter layer.
- Add an orchestrator service for long-running sessions and scheduling.
- Add autoscaling rules based on `hls:segment_publish_rate` and worker queue length.


Appendix A — Concrete Tech Stack Recommendation

- Backend API: Python 3.12, FastAPI, Uvicorn
- Workers: Python asyncio + aioredis / redis-py + concurrent.futures or multiprocessing for CPU bound tasks
- Queue/Bus: Redis Streams or RabbitMQ
- Video generation: Containerized InfiniteTalk binary, GPU nodes if supported
- HLS remux: ffmpeg in a dedicated container
- Frontend: Next.js (App Router), hls.js
- CI/CD: GitHub Actions (unit tests, container build, image publish)
- Monitoring: Prometheus + Grafana + OpenTelemetry


Appendix B — Example File Layout (monorepo)

- api/
  - src/app/main.py
  - src/app/hls_writer.py
  - src/app/llm_integration.py
  - Dockerfile
- workers/tts/
  - tts_worker.py
- workers/video/
  - video_worker.py
- infra/
  - docker-compose.yml
  - k8s/
- web/
  - next app


Appendix C — Quick checklist to verify current repo vs blueprint

- [ ] Are heavy jobs (video generation) extracted to workers? (If not, do it next.)
- [ ] Are messages passed via durable queue? (Repo currently couples via in-process calls.)
- [ ] Is ffmpeg/idle loop managed as separate process/container? (Repo already uses global FFmpeg; confirm it's robust.)
- [ ] Is observability (metrics, structured logs) present? (Add it.)
- [ ] Are secrets loaded securely and not committed? (Use env and secret manager.)



Next steps I recommend you take now
- Stabilize the global ffmpeg process in one small change: run it in a system-managed process (container or systemd) and add a healthcheck/restart logic in `src/app/main.py` so the API doesn't assume ffmpeg is always present.
- Implement a single Redis Stream as the first decoupling step and move RollingWav and video generation to worker processes; this will immediately improve reliability.
- Add structured logs and Prometheus metrics to `src/app/main.py` and workers.


If you'd like, I can:
- Create a `docker-compose.yml` skeleton matching the recommended layout.
- Begin extracting the video generator into a worker and wiring Redis Streams.
- Create the initial `Project_Architecture_Blueprint.md` (this file) into the repo (done).


---

End of blueprint.
