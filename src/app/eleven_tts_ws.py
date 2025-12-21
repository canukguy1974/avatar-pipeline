import os, json, asyncio, websockets, base64, logging, subprocess

ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "gs0tAILXbY5DNrJrsM6F")
ELEVEN_MODEL_ID = os.getenv("ELEVEN_MODEL_ID", "eleven_flash_v2_5")

def _mp3_bytes_to_pcm16k_s16le(mp3_bytes: bytes) -> bytes:
    """
    Convert MP3 bytes -> PCM s16le 1ch 16k using ffmpeg via stdin/stdout.
    Zero-copy through temp files; fast enough for streaming.
    """
    # Start ffmpeg process that reads from stdin and writes raw s16le to stdout
    p = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "mp3", "-i", "pipe:0",
         "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    out, _ = p.communicate(input=mp3_bytes)
    return out

async def elevenlabs_stream(tokens_async_iter, roller):
    """
    Connects to TTS WebSocket, streams text chunks, receives MP3 audio chunks,
    converts them to PCM16k s16le, and writes into RollingWav.
    """
    api_key = os.environ["ELEVENLABS_API_KEY"]
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}/stream-input?model_id={ELEVEN_MODEL_ID}"

    logging.info(f"elevenlabs TTS ws -> {uri}")

    __all__ = ["elevenlabs_stream"]

    # NOTE: No headers. Key is in the first JSON we send.
    async with websockets.connect(uri, open_timeout=10, ping_interval=20) as ws:
        # Initialize connection (set key + optional voice/generation settings)
        await ws.send(json.dumps({
            "text": " ",  # prime the buffer with a space
            "xi_api_key": api_key,
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.8,
                "use_speaker_boost": False
            },
            "generation_config": {
                # balances latency vs quality; tweak later
                "chunk_length_schedule": [120, 160, 250, 290]
            }
        }))

        # Start a reader task that pulls audio chunks → PCM → roller
        async def reader():
            try:
                async for msg in ws:
                    data = json.loads(msg)
                    # TTS WS returns base64-encoded MP3 chunks under "audio"
                    b64 = data.get("audio")
                    if b64:
                        mp3_bytes = base64.b64decode(b64)
                        pcm = _mp3_bytes_to_pcm16k_s16le(mp3_bytes)
                        if pcm:
                            # Your helper that accepts s16le @16k:
                            roller.write_pcm16k_s16le(pcm)
                    if data.get("isFinal"):
                        break
            except Exception:
                logging.exception("elevenlabs reader failed")

        reader_task = asyncio.create_task(reader())

        # Stream LLM tokens as they arrive
        async for t in tokens_async_iter:
            await ws.send(json.dumps({"text": t}))
        # Signal end of input to force flush
        await ws.send(json.dumps({"text": ""}))

        await reader_task