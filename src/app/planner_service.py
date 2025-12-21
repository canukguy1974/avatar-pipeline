import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# CONFIG (tune these)
REDIS_URL = "redis://localhost:6379/0"
LOOKAHEAD_MS = 400         # how far ahead planner timestamps start_time (ms)
CHUNK_TOKEN_LIMIT = 12     # send chunk every N tokens or after CHUNK_TIME_MS
CHUNK_TIME_MS = 150        # or after this many ms
STREAM_PREFIX = "stream:planner:"  # final stream: stream:planner:{session_id}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis client (reused)
redis = None


@app.on_event("startup")
async def startup_event():
    global redis
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)


@app.on_event("shutdown")
async def shutdown_event():
    global redis
    if redis:
        await redis.close()


# -------------------------
# Mock LLM token streamer
# -------------------------
async def mock_llm_stream(prompt: str) -> AsyncGenerator[str, None]:
    """Mock token stream. Replace with real LLM streaming generator."""
    # naive tokenization for demo
    tokens = prompt.split() + ["<END>"]
    for t in tokens:
        await asyncio.sleep(0.06)  # simulate latency per token
        yield t


# -------------------------
# Create chunk JSON and publish
# -------------------------
def make_chunk(session_id: str, seq: int, text_so_far: str, tokens: list, now_ms: int, duration_ms: int = 0):
    """
    Chunk schema:
    {
      "session_id": "...",
      "seq": 1,
      "start_time_ms": 169...,  # epoch ms client should target
      "duration_ms": 200,
      "text": "partial text",
      "tokens": ["...","..."],
      "intents": {...},            # optional structured intent metadata
      "meta": {...}
    }
    """
    start_time_ms = int(now_ms + LOOKAHEAD_MS)
    chunk = {
        "session_id": session_id,
        "seq": seq,
        "start_time_ms": start_time_ms,
        "duration_ms": duration_ms,
        "text": text_so_far,
        "tokens": tokens,
        "intents": {},  # planner can fill structured intent here (emotion, emphasis, action)
        "meta": {
            "generated_at_ms": int(now_ms),
            "publisher_id": "planner:" + str(uuid.uuid4())[:8]
        }
    }
    return chunk


# -------------------------
# WebSocket endpoint
# -------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_state = {}
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            action = data.get("action")
            if action == "start":
                session_id = data.get("session_id") or str(uuid.uuid4())
                prompt = data.get("prompt", "")
                role = data.get("role", "user")
                # save session
                session_state["session_id"] = session_id
                session_state["prompt"] = prompt
                # launch planner task
                asyncio.create_task(handle_planner_stream(ws, session_id, prompt))
                await ws.send_text(json.dumps({"status": "started", "session_id": session_id}))
            elif action == "stop":
                # not implemented: signal to stop stream
                await ws.send_text(json.dumps({"status": "stop-not-implemented"}))
            else:
                await ws.send_text(json.dumps({"error": "unknown action"}))
    except WebSocketDisconnect:
        print("WebSocket disconnected")


async def handle_planner_stream(ws: WebSocket, session_id: str, prompt: str):
    """
    Stream tokens, chunk them, publish to Redis stream and echo to websocket
    """
    seq = 0
    token_buffer = []
    text_accum = ""
    last_flush = time.time() * 1000

    async for token in mock_llm_stream(prompt):
        now_ms = int(time.time() * 1000)
        token_buffer.append(token)
        text_accum = (text_accum + " " + token).strip()
        elapsed_ms = now_ms - last_flush

        if len(token_buffer) >= CHUNK_TOKEN_LIMIT or elapsed_ms >= CHUNK_TIME_MS or token == "<END>":
            seq += 1
            chunk = make_chunk(session_id=session_id, seq=seq, text_so_far=text_accum, tokens=token_buffer, now_ms=now_ms)
            # publish to redis stream
            stream_key = STREAM_PREFIX + session_id
            try:
                await redis.xadd(stream_key, {"chunk": json.dumps(chunk)}, id="*")
            except Exception as e:
                # log but don't crash
                print("Redis publish failed:", e)
            # echo to websocket (so client sees incremental text too)
            try:
                await ws.send_text(json.dumps({"type": "planner_chunk", "chunk": chunk}))
            except Exception:
                # client gone - stop
                return
            # reset buffer
            token_buffer = []
            last_flush = int(time.time() * 1000)

        if token == "<END>":
            # finalization: publish a final marker with metadata if needed
            seq += 1
            final_chunk = make_chunk(session_id=session_id, seq=seq, text_so_far=text_accum, tokens=[], now_ms=int(time.time()*1000), duration_ms=0)
            final_chunk["meta"]["final"] = True
            await redis.xadd(STREAM_PREFIX + session_id, {"chunk": json.dumps(final_chunk)}, id="*")
            try:
                await ws.send_text(json.dumps({"type":"planner_end", "chunk": final_chunk}))
            except Exception:
                pass
            return