# src/app/hls_writer.py
import subprocess, mimetypes, contextlib, math
from pathlib import Path
from typing import List, Tuple

class HLSWriter:
    def __init__(self, hls_dir: Path, window: int = 6, default_td: float = 2.0, delete_old: bool = True):
        self.hls_dir = Path(hls_dir)
        self.hls_dir.mkdir(parents=True, exist_ok=True)
        self.window = window
        self.default_td = default_td
        self.delete_old = delete_old
        self._segments: List[Tuple[str, float, bool]] = []
        self._pending_discontinuity = False
        self._current_source = "idle"
        self._media_sequence = 0

        # MIME so FastAPI serves them correctly
        mimetypes.add_type("application/vnd.apple.mpegurl", ".m3u8")
        mimetypes.add_type("video/mp4", ".m4s")

        self._write_manifest()

    def _write_manifest(self):
        # Don’t write EXT-X-MAP until init.mp4 exists
        out = []
        out.append("#EXTM3U")
        out.append("#EXT-X-VERSION:7")
        # Compute target duration as the ceiling of the largest segment duration
        max_seg = max((d for (_, d, _) in self._segments), default=self.default_td)
        target_td = max(1, int(math.ceil(max(self.default_td, max_seg))) )
        out.append("#EXT-X-TARGETDURATION:%d" % target_td)
        out.append(f"#EXT-X-MEDIA-SEQUENCE:{self._media_sequence}")
        if (self.hls_dir / "init.mp4").exists():
            out.append('#EXT-X-MAP:URI="init.mp4"')
        for (name, dur, discont) in self._segments[-self.window:]:
            if discont:
                out.append("#EXT-X-DISCONTINUITY")
            out.append(f"#EXTINF:{dur:.3f},")
            out.append(name)
        (self.hls_dir / "manifest.m3u8").write_text("\n".join(out) + "\n", encoding="utf-8")
    
    def add_segment(self, filename: str, duration: float):
        self._segments.append((filename, duration, self._pending_discontinuity))
        self._pending_discontinuity = False
        while self.delete_old and len(self._segments) > self.window:
            old_name, *_ = self._segments.pop(0)
            with contextlib.suppress(FileNotFoundError):
                (self.hls_dir / old_name).unlink()
            self._media_sequence += 1
        self._write_manifest()
    def force_discontinuity(self):
        self._pending_discontinuity = True

    def switch_source(self, new_source: str, *, force: bool = False):
        if force or new_source != self._current_source:
            self._current_source = new_source
            self._pending_discontinuity = True

    def start_idle_segmentation(self, idle_mp4: Path):
        """FFmpeg: loop idle.mp4 forever → idle_%06d.m4s (copy codec, no re-encoding).
        
        Note: We use finite duration (-to flag) and restart FFmpeg in a loop to achieve
        infinite looping with proper segment output. This is because FFmpeg's -stream_loop
        with copy codec doesn't produce segments until the duration is known.
        """
        self.hls_dir.mkdir(parents=True, exist_ok=True)
        # don't start another ffmpeg if one is already running
        proc = getattr(self, '_idle_proc', None)
        try:
            if proc and proc.poll() is None:
                return
        except Exception:
            pass
        
        # Use absolute path for input
        idle_mp4_abs = idle_mp4.resolve()
        hls_dir_abs = self.hls_dir.resolve()
        
        # Use a 30-second duration (multiple of 3s segments) so FFmpeg will actually produce output
        # This is much longer than a single video playback, so it feels like infinite looping
        output_duration = "30s"
        
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-stream_loop", "10",  # Loop the input 10 times (should be plenty)
            "-i", str(idle_mp4_abs),
            "-t", output_duration,  # Limit total output duration
            "-c:v", "copy", "-c:a", "copy",
            "-f", "hls",
            "-hls_segment_type", "fmp4",
            "-hls_time", "3",
            "-hls_list_size", "8",
            "-hls_flags", "delete_segments+independent_segments",
            "-hls_fmp4_init_filename", "init.mp4",  # Relative to cwd
            "-hls_segment_filename", "idle_%06d.m4s",  # Relative to cwd
            "idle.m3u8",  # Relative to cwd
        ]
        print(f"DEBUG: Starting FFmpeg from {hls_dir_abs}", flush=True)
        print(f"DEBUG: FFmpeg will loop and produce ~{int(30/3)} segments", flush=True)
        # Run FFmpeg from the HLS directory so relative paths work correctly
        self._idle_proc = subprocess.Popen(cmd, cwd=str(hls_dir_abs))

    def stop_idle_segmentation(self):
        """Stop the FFmpeg idle segmentation process and remove idle segments/playlist."""
        try:
            proc = getattr(self, '_idle_proc', None)
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        except Exception:
            pass
        # Remove idle playlist and segments to free space
        try:
            for p in list(self.hls_dir.glob('idle_*.m4s')):
                with contextlib.suppress(Exception):
                    p.unlink()
            with contextlib.suppress(Exception):
                (self.hls_dir / 'idle.m3u8').unlink()
        except Exception:
            pass
