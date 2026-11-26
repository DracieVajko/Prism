# FFmpeg recipes (flags only)

Lenswright does **not** bundle ffmpeg as a tool. Media processing on Android runs
on the platform's `MediaCodec` / `MediaMuxer` / `MediaExtractor` (+ LAME for
MP3). On the PC side, ffmpeg is an **optional external binary** invoked with
the flags below — nothing in the codebase requires it except `ffmpeg_pre.py`
(audio normalization) and the `/v1/ffmpeg/*` endpoints, which fall back to
passthrough when ffmpeg is absent.

All flags are cross-platform (Windows/Linux/macOS). Replace `INPUT`/`OUTPUT`
with your files.

## Audio

### Normalize to 16 kHz mono WAV (Whisper input)
```
ffmpeg -y -i INPUT -ar 16000 -ac 1 OUTPUT.wav
```

### Extract audio track (AAC → M4A, no re-encode)
```
ffmpeg -y -i INPUT -vn -c:a copy OUTPUT.m4a
```

### Convert audio to AAC
```
ffmpeg -y -i INPUT -c:a aac -b:a 192k OUTPUT.m4a
```

### Convert audio to MP3 (LAME)
```
ffmpeg -y -i INPUT -c:a libmp3lame -b:a 192k OUTPUT.mp3
```

### Trim audio/video by time
```
ffmpeg -y -ss START -to END -i INPUT -c copy OUTPUT
```

## Video

### Remux to MP4 (no re-encode)
```
ffmpeg -y -i INPUT -c copy OUTPUT.mp4
```

### Compress to small MP4 (H.264)
```
ffmpeg -y -i INPUT -c:v libx264 -crf 32 -preset slow -c:a aac -b:a 96k OUTPUT.mp4
```

### High-quality H.264
```
ffmpeg -y -i INPUT -c:v libx264 -crf 18 -preset slow -c:a aac -b:a 192k OUTPUT.mp4
```

### Fast H.264
```
ffmpeg -y -i INPUT -c:v libx264 -crf 28 -preset veryfast -c:a aac -b:a 128k OUTPUT.mp4
```

### H.265/HEVC
```
ffmpeg -y -i INPUT -c:v libx265 -crf 28 -preset medium -c:a aac -b:a 128k OUTPUT.mp4
```

### WebM (VP9 + Opus)
```
ffmpeg -y -i INPUT -c:v libvpx-vp9 -crf 32 -b:v 0 -c:a libopus -b:a 128k OUTPUT.webm
```

## Image

### Convert image format / quality
```
ffmpeg -y -i INPUT -q:v 2 OUTPUT.jpg
```

## Info

### Media info (ffprobe)
```
ffprobe -v quiet -print_format json -show_format -show_streams INPUT
```

## Notes
- These are the flags used by `pc/unified/ffmpeg_tools.py` presets. Keep ffmpeg on
  `PATH`; Lenswright detects it via `shutil.which("ffmpeg")` and degrades gracefully
  (passthrough) when absent.
- Android: no ffmpeg needed — `MediaTools` (MediaCodec) covers the same ops.