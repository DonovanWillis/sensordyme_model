# Sample data for benchmarking

Drop test video files (`.mp4`, `.mov`, `.avi`) in this directory, then point
`run_benchmarks.py` at one with `--video`:

```bash
python benchmarks/run_benchmarks.py --video benchmarks/sample_data/driving_clip.mp4 --minutes 2
```

If you omit `--video`, the benchmark runs against the live camera (Pi camera
or Mac webcam, auto-detected) instead.

Suggested clips per module:
- `driver_attention`: a face-on video of someone driving or sitting at a desk,
  ideally including a few natural blinks and head turns.
- `occupancy`: a wide shot of a room with people entering/leaving.
- `fall_detection`: a clip of a person walking and then sitting/lying down.

Files placed here are gitignored — this directory only ships the README.
