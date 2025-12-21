# src/app/controller.py
from pathlib import Path
from typing import AsyncIterator, Tuple
from app.hls_writer import HLSWriter

# These should be provided by main.py (we inject them)
# idle_segment_stream: () -> AsyncIterator[Tuple[Path, float]]
# live_segment_stream: () -> AsyncIterator[Tuple[Path, float]]

async def run_controller(
    hls: HLSWriter,
    idle_segment_stream,
    live_segment_stream,
    ws=None,  # optional WebSocket to notify the client
):
    # Idle pre-roll until live starts
    hls.switch_source("idle")
    async for seg_path, seg_dur in idle_segment_stream():
        hls.add_segment(seg_path.name, seg_dur)
        if ws:
            await ws.send_json({"type": "video_segment", "uri": f"/hls/{seg_path.name}", "dur": seg_dur})

    # Flip to live when you’re ready (you decide when to stop idle above)
    hls.switch_source("live")
    async for seg_path, seg_dur in live_segment_stream():
        hls.add_segment(seg_path.name, seg_dur)
        if ws:
            await ws.send_json({"type": "video_segment", "uri": f"/hls/{seg_path.name}", "dur": seg_dur})
