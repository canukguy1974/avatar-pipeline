# src/app/hls_writer.py
import subprocess, mimetypes, contextlib, math, os
from pathlib import Path
from typing import List, Tuple

class HLSWriter:
    def __init__(self, hls_dir: Path, window: int = 20, default_td: float = 6.0, delete_old: bool = True):
        self.hls_dir = Path(hls_dir)
        self.hls_dir.mkdir(parents=True, exist_ok=True)
        self.window = window
        self.default_td = default_td
        self.delete_old = delete_old
        
        # Tuple: (filename, duration, discontinuity, map_file)
        self._segments: List[Tuple[str, float, bool, str]] = []
        self._pending_discontinuity = False
        self._current_source = "idle"
        self._media_sequence = 0

        # MIME so FastAPI serves them correctly
        mimetypes.add_type("application/vnd.apple.mpegurl", ".m3u8")
        mimetypes.add_type("video/mp4", ".m4s")

        self._write_manifest()

    def _write_manifest(self):
        out = []
        out.append("#EXTM3U")
        out.append("#EXT-X-VERSION:7")
        out.append("#EXT-X-INDEPENDENT-SEGMENTS")
        
        # Compute target duration accurately
        max_dur = max((d for (_, d, _, _) in self._segments), default=self.default_td)
        target_td = int(math.ceil(max(self.default_td, max_dur)))
        out.append(f"#EXT-X-TARGETDURATION:{target_td}")
        
        out.append(f"#EXT-X-MEDIA-SEQUENCE:{self._media_sequence}")
        
        # Calculate DISCONTINUITY-SEQUENCE
        # It should represent the total number of discontinuities that have FALLEN OFF the sliding window.
        # But for simpler live streaming, we can just sum all DISCONTINUITY tags ever popped.
        # However, we don't have global history, so we'll just track it with a counter.
        if not hasattr(self, '_discontinuity_sequence'):
            self._discontinuity_sequence = 0
        out.append(f"#EXT-X-DISCONTINUITY-SEQUENCE:{self._discontinuity_sequence}")

        last_map = None
        # We only write the segments in the window
        for (name, dur, discont, map_file) in self._segments:
            if discont:
                out.append("#EXT-X-DISCONTINUITY")
                # Player REQUIRES map re-declaration after discontinuity for fMP4
                if map_file:
                    out.append(f'#EXT-X-MAP:URI="{map_file}"')
                    last_map = map_file
            elif map_file and map_file != last_map:
                out.append(f'#EXT-X-MAP:URI="{map_file}"')
                last_map = map_file
            
            out.append(f"#EXTINF:{dur:.6f},")
            out.append(name)

        tmp_manifest = self.hls_dir / "manifest.m3u8.tmp"
        tmp_manifest.write_text("\n".join(out) + "\n", encoding="utf-8")
        try:
            tmp_manifest.replace(self.hls_dir / "manifest.m3u8")
        except Exception:
            with contextlib.suppress(Exception):
                os.remove(str(self.hls_dir / "manifest.m3u8"))
            tmp_manifest.replace(self.hls_dir / "manifest.m3u8")

    def add_segment(self, filename: str, duration: float, map_file: str = "init.mp4"):
        self._segments.append((filename, duration, self._pending_discontinuity, map_file))
        self._pending_discontinuity = False
        
        while len(self._segments) > self.window:
            popped = self._segments.pop(0)
            p_name, p_dur, p_discont, p_map = popped
            if self.delete_old:
                with contextlib.suppress(Exception):
                    (self.hls_dir / p_name).unlink()
            self._media_sequence += 1
            if p_discont:
                if not hasattr(self, '_discontinuity_sequence'):
                    self._discontinuity_sequence = 0
                self._discontinuity_sequence += 1
        
        self._write_manifest()

    def force_discontinuity(self):
        self._pending_discontinuity = True

    def switch_source(self, new_source: str, *, force: bool = False):
        if force or new_source != self._current_source:
            self._current_source = new_source
            self._pending_discontinuity = True

    def start_idle_segmentation(self, idle_mp4: Path):
        self.hls_dir.mkdir(parents=True, exist_ok=True)
        # don't start another ffmpeg if one is already running
        proc = getattr(self, '_idle_proc', None)
        try:
            if proc and proc.poll() is None:
                return
        except Exception:
            pass
        
        idle_mp4_abs = idle_mp4.resolve()
        hls_dir_abs = self.hls_dir.resolve()
        
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "verbose", "-stats",
            "-re", 
            "-stream_loop", "-1",
            "-i", str(idle_mp4_abs),
            "-c:v", "copy", "-c:a", "copy",
            "-f", "hls",
            "-hls_segment_type", "fmp4",
            "-hls_time", "5",
            "-hls_list_size", "30",
            "-hls_flags", "delete_segments+independent_segments",
            "-hls_fmp4_init_filename", "init.mp4",
            "-hls_segment_filename", "idle_%06d.m4s",
            "idle.m3u8",
        ]
        print(f"DEBUG: Starting FFmpeg from {hls_dir_abs}", flush=True)
        self._idle_proc = subprocess.Popen(cmd, cwd=str(hls_dir_abs))

    def stop_idle_segmentation(self):
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
        try:
            for p in list(self.hls_dir.glob('idle_*.m4s')):
                with contextlib.suppress(Exception):
                    p.unlink()
            with contextlib.suppress(Exception):
                (self.hls_dir / 'idle.m3u8').unlink()
        except Exception:
            pass
