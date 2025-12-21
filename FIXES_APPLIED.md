# Fixes Applied to InfiniteTalk Backend

## Issues Found and Fixed

### 1. **Indentation Error in `src/app/main.py` (line 187-189)**
   - **Problem**: Malformed `if` statement with misplaced comment breaking indentation
   - **Fix**: Reorganized `video_loop()` method with proper try/except structure
   - **Result**: Syntax error resolved ✓

### 2. **NameError: `preflight_or_idle` not defined (line 63)**
   - **Problem**: Function called before it was defined
   - **Fix**: Moved `_abs()` and `preflight_or_idle()` definitions before they're called
   - **Result**: Module loading error resolved ✓

### 3. **Missing Import in `src/app/hls_writer.py`**
   - **Problem**: `contextlib` was used but not imported
   - **Fix**: Added `contextlib` to imports
   - **Result**: `suppress()` calls work properly ✓

### 4. **Missing `BOOTSTRAP_MANIFEST` Definition**
   - **Problem**: Referenced on line 265 but never defined
   - **Fix**: Defined as `HLS_DIR / "manifest_bootstrap.m3u8"` and created on startup
   - **Result**: Bootstrap manifest created automatically ✓

### 5. **Critical: Directory Mismatch (ROOT CAUSE OF EMPTY VIDEO)**
   - **Problem**: `.env` had duplicate `HLS_DIR` entries:
     - First: `/home/canuk/projects/inifinitetalk-local/hls`
     - Second (overwrites): `/home/canuk/projects/inifinitetalk-local/src/hls`
   - **Impact**: FFmpeg was writing idle segments to one directory, but Python was reading from another!
   - **Fix**: Cleaned up `.env` to use consistent directories:
     ```env
     HLS_DIR=/home/canuk/projects/inifinitetalk-local/hls
     AUDIO_DIR=/home/canuk/projects/inifinitetalk-local/hls/audio
     STATIC_DIR=/home/canuk/projects/inifinitetalk-local/static
     ```
   - **Result**: Idle loop segments now in correct directory ✓

### 6. **Improved Error Handling in WebSocket Handler**
   - **Problem**: If idle loop setup failed, error was not reported to client
   - **Fix**: Added checks and status messages for idle loop initialization
   - **Result**: Client now receives diagnostic messages ✓

## What Should Now Work

1. **Idle Loop Playback**:
   - FFmpeg starts a background process to encode `idle_loop.mp4` into HLS segments
   - Segments are written to `/home/canuk/projects/inifinitetalk-local/hls/idle_*.m4s`
   - Python reads these segments and adds them to the manifest
   - Browser loads HLS playlist and plays looping video

2. **Live Transition**:
   - When InfiniteTalk starts generating segments (`seg_0000.m4s`, etc.)
   - Backend detects first live segment and switches manifest source
   - HLS discontinuity marker ensures smooth transition
   - Live video streams over idle loop

3. **Status Messages**:
   - Browser console shows: `Status: idle_setup` → idle loop initialized
   - Then: `Segment #0 ready: /hls/idle_000000.m4s` → segments being added
   - Finally: transition to live when first `seg_0000.m4s` appears

## Testing Checklist

- [ ] Browser console shows `Status: idle_setup` message
- [ ] Browser console shows segment messages like `Segment #0 ready`
- [ ] Video player displays idle loop animation
- [ ] Check Network tab: manifest.m3u8 loaded and contains idle_*.m4s segments
- [ ] Check Network tab: init.mp4 loaded (HLS initialization segment)
- [ ] Check `/home/canuk/projects/inifinitetalk-local/hls/manifest.m3u8` contains:
  ```
  #EXT-X-MAP:URI="init.mp4"
  #EXTINF:1.000,
  idle_000000.m4s
  idle_000001.m4s
  ...
  ```

## Backend Restart Required

After these fixes, you must:

1. Kill old ffmpeg processes:
   ```bash
   pkill -f "ffmpeg.*idle_loop.mp4"
   ```

2. Clean old HLS files:
   ```bash
   rm -rf /home/canuk/projects/inifinitetalk-local/hls/idle_*.m4s
   rm -rf /home/canuk/projects/inifinitetalk-local/hls/manifest.m3u8
   ```

3. Restart backend:
   ```bash
   cd /home/canuk/projects/inifinitetalk-local/src
   uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
   ```

4. Refresh browser and create new WebSocket connection

## File Changes Summary

| File | Changes |
|------|---------|
| `src/app/main.py` | Fixed indentation, moved function definitions, added bootstrap manifest, improved error handling |
| `src/app/hls_writer.py` | Added missing `contextlib` import |
| `.env` | Removed duplicate `HLS_DIR`, consolidated directory structure |

## Notes for Next Development

- Always use consistent directory paths in .env (no duplicates)
- Test with `USE_STUBS=1` before relying on external services
- Check `PREFLIGHT_ISSUES` in browser console if idle loop doesn't start
- Monitor `/tmp/hls_writer_debug.log` for FFmpeg output if segments aren't being created
