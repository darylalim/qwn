"""Cut the demo video from a recording (PLAN.md → Demo video → Assemble step).

    uv run python scripts/demo_video/assemble.py <recording folder> --music PATH

ffmpeg (and Pillow for the zoom frames), no models: seconds per run, so caption and timing
edits in timeline.toml need no new recording. Writes dist/video/: qwn-demo.mp4 (music and
the voice shot's audio, -14 LUFS), qwn-demo-silent.mp4 (the same video stream, no audio),
qwn-demo.srt and qwn-demo-thumbnail.png (1280x720). Without --music, qwn-demo.mp4 carries
the voice shot's audio only (for review); the music version needs the file.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))

import timeline  # noqa: E402

LUFS, TRUE_PEAK, LRA = -14.0, -1.5, 11.0
RATE = 48_000


def ffmpeg(*args: str, capture: bool = False) -> str:
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", *args]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        sys.exit(f"ffmpeg failed ({run.returncode}):\n{' '.join(cmd)}\n{run.stderr[-3000:]}")
    return run.stderr if capture else ""


def link_frames(folder: Path, plan: timeline.Plan, cut: Path) -> None:
    """The output frames as a numbered sequence of hard links to the recorded JPEGs."""
    if cut.exists():
        shutil.rmtree(cut)
    cut.mkdir(parents=True)
    for n, (shot, index) in enumerate(plan.frames):
        src = folder / "frames" / shot / f"{index:05d}.jpg"
        if n in plan.zooms:
            zoomed(src, plan.zooms[n], cut / f"{n:06d}.jpg")
        else:
            (cut / f"{n:06d}.jpg").hardlink_to(src)


def zoomed(src: Path, box: tuple[float, float, float, float], out: Path) -> None:
    """The frame's crop box scaled back to full size, with sub-pixel placement (no jitter)."""
    from PIL import Image

    with Image.open(src) as im:
        frame = im.transform(im.size, Image.Transform.EXTENT, box, Image.Resampling.BICUBIC)
        frame.save(out, quality=95)


def encode_video(folder: Path, plan: timeline.Plan, out: Path) -> None:
    cut = folder / "cut"
    link_frames(folder, plan, cut)
    images, graph = timeline.overlay_filter(plan)
    script = folder / "video.filter"
    script.write_text(graph)
    inputs = [arg for img in images for arg in ("-i", str(folder / img))]
    ffmpeg(
        "-framerate", str(plan.fps), "-start_number", "0", "-i", str(cut / "%06d.jpg"),
        *inputs,
        "-/filter_complex", str(script), "-map", "[v]",
        "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-crf", "18",
        "-preset", "slow", "-r", str(plan.fps), "-an", "-movflags", "+faststart", str(out),
    )  # fmt: skip


def mix_audio(folder: Path, plan: timeline.Plan, music: Path | None, out: Path) -> None:
    """The music under its volume curve plus the voice cues, two-pass loudness-normalised."""
    dur = f"{plan.duration:.4f}"
    inputs: list[str] = []
    parts, labels = [], []
    if music is not None:
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        expr = timeline.volume_expr(plan.music)
        parts.append(
            f"[0:a]aresample={RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"atrim=0:{dur},volume='{expr}':eval=frame[m]"
        )
        labels.append("[m]")
    n = len(labels)  # ffmpeg input index of the next cue
    for cue in plan.cues:
        inputs += ["-i", str(folder / cue.audio)]
        ms = round(cue.start * 1000)
        parts.append(
            f"[{n}:a]aresample={RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"adelay={ms}|{ms}[c{n}]"
        )
        labels.append(f"[c{n}]")
        n += 1
    if not labels:
        sys.exit("nothing to mix: no music and no voice cues")
    parts.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest,"
        f"apad,atrim=0:{dur}[mix]"
    )
    script = folder / "audio.filter"
    script.write_text(";\n".join(parts))
    raw = folder / "mix.wav"
    ffmpeg(*inputs, "-/filter_complex", str(script), "-map", "[mix]", "-ar", str(RATE), str(raw))
    target = f"I={LUFS}:TP={TRUE_PEAK}:LRA={LRA}"
    log = ffmpeg(
        "-i", str(raw), "-af", f"loudnorm={target}:print_format=json", "-f", "null", "-",
        capture=True,
    )  # fmt: skip
    m = json.loads(log[log.rindex("{") : log.rindex("}") + 1])
    measured = (
        f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
        f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true"
    )
    ffmpeg("-i", str(raw), "-af", f"loudnorm={target}:{measured}", "-ar", str(RATE), str(out))


def loudness(path: Path) -> float:
    log = ffmpeg("-i", str(path), "-af", "ebur128", "-f", "null", "-", capture=True)
    found = re.findall(r"I:\s+(-?[\d.]+) LUFS", log)
    return float(found[-1]) if found else float("nan")


def probe(path: Path) -> str:
    run = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,profile,width,height,pix_fmt,r_frame_rate,sample_rate,"
         "nb_frames:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    info = json.loads(run.stdout)
    rows = []
    for s in info["streams"]:
        if s["codec_type"] == "video":
            rows.append(
                f"video {s['codec_name']} {s.get('profile')} {s['width']}x{s['height']} "
                f"{s['pix_fmt']} {s['r_frame_rate']} fps, {s.get('nb_frames')} frames"
            )
        else:
            rows.append(f"audio {s['codec_name']} {s.get('sample_rate')} Hz")
    rows.append(f"duration {float(info['format']['duration']):.2f} s")
    return "; ".join(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("folder", type=Path, help="the record step's output")
    p.add_argument("--music", type=Path, default=None)
    p.add_argument("--timeline", type=Path, default=ROOT / "docs" / "video" / "timeline.toml")
    p.add_argument("--out", type=Path, default=ROOT / "dist" / "video")
    args = p.parse_args()
    folder = args.folder.resolve()
    if args.music is not None and not args.music.is_file():
        sys.exit(f"no music file at {args.music}")
    try:
        tl = timeline.load(args.timeline)
        plan = timeline.build_plan(tl, timeline.read_events(folder))
    except timeline.TimelineError as e:
        sys.exit(f"assemble: {e}")
    print(timeline.summary(plan, tl))
    args.out.mkdir(parents=True, exist_ok=True)

    video = args.out / "qwn-demo-silent.mp4"
    encode_video(folder, plan, video)
    (args.out / "qwn-demo.srt").write_text(timeline.srt(plan))

    audio = folder / "mix-norm.wav"
    mix_audio(folder, plan, args.music, audio)
    final = args.out / "qwn-demo.mp4"
    ffmpeg(
        "-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(RATE), "-movflags", "+faststart", str(final),
    )  # fmt: skip

    shot, index = plan.thumbnail
    frame = folder / "frames" / shot / f"{index:05d}.jpg"
    ffmpeg(
        "-i", str(frame), "-i", str(folder / "overlays" / "title.png"), "-filter_complex",
        "[0:v][1:v]overlay=0:0,scale=1280:720:flags=lanczos", "-frames:v", "1",
        str(args.out / "qwn-demo-thumbnail.png"),
    )  # fmt: skip

    for path in (final, video):
        print(f"{path.name}: {probe(path)}")
    print(f"qwn-demo.mp4 integrated loudness: {loudness(final):.1f} LUFS (target {LUFS:g})")
    if args.music is None:
        print("no --music: qwn-demo.mp4 has the voice shot's audio only")


if __name__ == "__main__":
    main()
