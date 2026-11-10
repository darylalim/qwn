"""The demo video's timeline (PLAN.md → Demo video): parse and check `docs/video/timeline.toml`,
then turn it and a recording's `events.json` into the cut that `assemble.py` encodes.

Stdlib only and pure, so its tests need no ffmpeg, Chrome or models. Times in a recording are
seconds on each shot's own frame clock (0 = its first frame); times in a `Plan` are seconds of
output video.

A shot plays at 1x except its waits (an event pair such as "sent" -> "answer"), which play at
their speed factor. Every sped-up stretch gets the label overlay for its whole length, and
`build_plan` checks that, so the video never hides a wait.
"""

import bisect
import json
import math
import tomllib
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

# The events each kind of shot logs, in the order they happen.
KINDS: dict[str, tuple[str, ...]] = {
    "chat": ("start", "sent", "answer", "end"),
    "refusal": ("start", "sent", "answer", "end"),
    "terminal": ("start", "sent", "answer", "end"),
    "open_page": ("start", "click", "dialog", "end"),
    "followup": ("start", "sent1", "answer1", "sent2", "answer2", "end"),
    "voice": ("start", "rec_start", "rec_stop", "answer", "audio", "audio_end", "end"),
    "library": ("start", "end"),
    "card": ("start", "end"),
}
WITH_QUERY = {"chat", "refusal", "terminal", "followup", "voice"}
# Stretches whose sound plays at its logged time: they must play at 1x.
AUDIO_SPANS = {"voice": (("rec_start", "rec_stop"), ("audio", "audio_end"))}


class TimelineError(ValueError):
    pass


@dataclass(frozen=True)
class Wait:
    start: str
    end: str
    speed: float


@dataclass(frozen=True)
class Shot:
    id: str
    kind: str
    caption: str | None = None
    query: str | None = None
    tag: str | None = None
    lead_s: float = 0.0
    mid_s: float = 0.0
    hold_s: float = 0.0
    waits: tuple[Wait, ...] = ()
    lines: tuple[str, ...] = ()
    url: str | None = None
    zoom: float = 1.0  # open_page: zoom on the passage once the dialog is ready
    zoom_ease_s: float = 0.0


@dataclass(frozen=True)
class Music:
    level_db: float
    fade_in_s: float
    duck_s: float
    fade_out_s: float


@dataclass(frozen=True)
class Timeline:
    width: int
    height: int
    fps: int
    min_s: float
    max_s: float
    type_delay_s: float
    music: Music
    thumbnail_shot: str
    title: str
    subtitle: str
    shots: tuple[Shot, ...]

    def shot(self, shot_id: str) -> Shot:
        return next(s for s in self.shots if s.id == shot_id)


def label_text(speed: float) -> str:
    return f"Sped up {speed:g}\N{MULTIPLICATION SIGN}"


def overlay_name(kind: str, key: str | float) -> str:
    """The PNG an overlay is drawn from, relative to the recording folder."""
    if kind == "label":
        return f"overlays/label-{key:g}x.png"
    return f"overlays/{kind}-{key}.png"


def overlays(tl: Timeline) -> dict[str, tuple[str, str]]:
    """Every overlay PNG the record step renders: {name: (style, text)}."""
    out: dict[str, tuple[str, str]] = {}
    for s in tl.shots:
        if s.caption:
            out[overlay_name("caption", s.id)] = ("caption", s.caption)
        if s.tag:
            out[overlay_name("tag", s.id)] = ("tag", s.tag)
        for w in s.waits:
            out[overlay_name("label", w.speed)] = ("label", label_text(w.speed))
    return out


def _num(d: dict, key: str, default: float = 0.0) -> float:
    v = d.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int | float) or v < 0:
        raise TimelineError(f"{key} must be a number >= 0, got {v!r}")
    return float(v)


def parse(text: str) -> Timeline:
    """A checked Timeline from the TOML text; TimelineError says what's wrong."""
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise TimelineError(f"not valid TOML: {e}") from None
    video, music, thumb = raw.get("video", {}), raw.get("music", {}), raw.get("thumbnail", {})
    shots = tuple(_shot(i, s) for i, s in enumerate(raw.get("shot", [])))
    tl = Timeline(
        width=int(_num(video, "width", 1920)),
        height=int(_num(video, "height", 1080)),
        fps=int(_num(video, "fps", 30)),
        min_s=_num(video, "min_s"),
        max_s=_num(video, "max_s", math.inf),
        type_delay_s=_num(video, "type_delay_s", 0.045),
        music=Music(
            level_db=float(music.get("level_db", -18.0)),
            fade_in_s=_num(music, "fade_in_s", 1.0),
            duck_s=_num(music, "duck_s", 0.5),
            fade_out_s=_num(music, "fade_out_s", 2.0),
        ),
        thumbnail_shot=str(thumb.get("shot", "")),
        title=str(thumb.get("title", "")),
        subtitle=str(thumb.get("subtitle", "")),
        shots=shots,
    )
    _check(tl)
    return tl


def load(path: Path) -> Timeline:
    return parse(path.read_text())


def _shot(i: int, s: dict) -> Shot:
    where = f"shot {i + 1} ({s.get('id', '?')})"
    kind = s.get("kind")
    if kind not in KINDS:
        raise TimelineError(f"{where}: kind must be one of {sorted(KINDS)}, got {kind!r}")
    known = {
        "id", "kind", "caption", "query", "tag", "lead_s", "mid_s", "hold_s", "waits", "lines",
        "url", "zoom", "zoom_ease_s",
    }  # fmt: skip
    if extra := set(s) - known:
        raise TimelineError(f"{where}: unknown keys {sorted(extra)}")
    waits = []
    for w in s.get("waits", []):
        if set(w) != {"from", "to", "speed"}:
            raise TimelineError(f"{where}: a wait needs exactly from, to and speed")
        speed = _num(w, "speed")
        if speed <= 1:
            raise TimelineError(f"{where}: a wait's speed must be above 1 (1x needs no wait)")
        waits.append(Wait(str(w["from"]), str(w["to"]), speed))
    try:
        return Shot(
            id=str(s["id"]),
            kind=kind,
            caption=s.get("caption"),
            query=s.get("query"),
            tag=s.get("tag"),
            lead_s=_num(s, "lead_s"),
            mid_s=_num(s, "mid_s"),
            hold_s=_num(s, "hold_s"),
            waits=tuple(waits),
            lines=tuple(s.get("lines", ())),
            url=s.get("url"),
            zoom=_num(s, "zoom", 1.0),
            zoom_ease_s=_num(s, "zoom_ease_s"),
        )
    except (KeyError, TimelineError) as e:
        raise TimelineError(f"{where}: {e}") from None


def _check(tl: Timeline) -> None:
    if not tl.shots:
        raise TimelineError("no shots")
    ids = [s.id for s in tl.shots]
    if len(set(ids)) != len(ids):
        raise TimelineError(f"shot ids must be unique: {ids}")
    if tl.thumbnail_shot not in ids:
        raise TimelineError(f"thumbnail shot {tl.thumbnail_shot!r} isn't a shot")
    for prev, s in zip((None, *tl.shots), tl.shots, strict=False):
        where = f"shot {s.id}"
        if (s.query is None) == (s.kind in WITH_QUERY):
            need = "needs" if s.kind in WITH_QUERY else "takes no"
            raise TimelineError(f"{where}: a {s.kind} shot {need} query")
        if s.kind != "card" and not s.caption:
            raise TimelineError(f"{where}: needs a caption")
        if s.kind == "card" and (not s.lines or not s.url):
            raise TimelineError(f"{where}: a card needs lines and url")
        if s.zoom < 1 or (s.zoom > 1 and s.kind != "open_page"):
            raise TimelineError(f"{where}: zoom must be >= 1, and only an open_page shot zooms")
        if s.kind == "open_page" and (prev is None or prev.kind != "chat"):
            raise TimelineError(f"{where}: open_page must follow a chat shot (it opens its answer)")
        order = KINDS[s.kind]
        spans = []
        for w in s.waits:
            if w.start not in order or w.end not in order:
                raise TimelineError(f"{where}: wait {w.start}->{w.end}: events are {order}")
            a, b = order.index(w.start), order.index(w.end)
            if a >= b:
                raise TimelineError(f"{where}: wait {w.start}->{w.end} runs backwards")
            spans.append((a, b))
        spans.sort()
        for (_, b1), (a2, _) in pairwise(spans):
            if a2 < b1:
                raise TimelineError(f"{where}: waits overlap")
        for x, y in AUDIO_SPANS.get(s.kind, ()):
            i, j = order.index(x), order.index(y)
            if any(a < j and i < b for a, b in spans):
                raise TimelineError(f"{where}: {x}->{y} has sound, so it must play at 1x")


# --- the cut -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """A stretch of one shot's recording [t0, t1) played at `speed`."""

    t0: float
    t1: float
    speed: float

    @property
    def out_s(self) -> float:
        return (self.t1 - self.t0) / self.speed


@dataclass(frozen=True)
class Overlay:
    image: str  # a full-frame transparent PNG, relative to the recording folder
    start: float
    end: float


@dataclass(frozen=True)
class Cue:
    audio: str  # a WAV, relative to the recording folder
    start: float


@dataclass
class Plan:
    fps: int
    frames: list[tuple[str, int]] = field(default_factory=list)  # (shot id, source frame index)
    overlays: list[Overlay] = field(default_factory=list)
    cues: list[Cue] = field(default_factory=list)
    captions: list[tuple[float, float, str]] = field(default_factory=list)
    spans: dict[str, tuple[float, float]] = field(default_factory=dict)  # shot -> output span
    sped: list[tuple[float, float, float]] = field(default_factory=list)  # (start, end, speed)
    music: list[tuple[float, float]] = field(default_factory=list)  # (time, gain) points
    zooms: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)  # crop box
    thumbnail: tuple[str, int] = ("", 0)

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps


def segments(shot: Shot, events: dict[str, float]) -> list[Segment]:
    """The shot's recording split at its waits; checks the events are all there, in order."""
    order = KINDS[shot.kind]
    missing = [e for e in order if e not in events]
    if missing:
        raise TimelineError(f"shot {shot.id}: events.json lacks {missing}")
    times = [events[e] for e in order]
    if any(b < a for a, b in pairwise(times)):
        raise TimelineError(
            f"shot {shot.id}: events out of order: {dict(zip(order, times, strict=True))}"
        )
    out, t = [], events["start"]
    for w in sorted(shot.waits, key=lambda w: events[w.start]):
        a, b = events[w.start], events[w.end]
        if a > t:
            out.append(Segment(t, a, 1.0))
        if b > a:
            out.append(Segment(a, b, w.speed))
        t = max(t, b)
    if events["end"] > t:
        out.append(Segment(t, events["end"], 1.0))
    return out


def out_time(segs: list[Segment], t: float) -> float:
    """Output seconds from the shot's start for recording time t."""
    acc = 0.0
    for s in segs:
        if t < s.t1 or s is segs[-1]:
            return acc + max(0.0, min(t, s.t1) - s.t0) / s.speed
        acc += s.out_s
    return acc


def src_time(segs: list[Segment], t_out: float) -> float:
    """Recording time for t_out output seconds into the shot (the inverse of out_time)."""
    acc = 0.0
    for s in segs:
        if t_out < acc + s.out_s:
            return s.t0 + (t_out - acc) * s.speed
        acc += s.out_s
    return segs[-1].t1


def ease(u: float) -> float:
    """Smoothstep: starts and ends at rest."""
    return u * u * (3 - 2 * u)


def zoom_box(
    width: int, height: int, focus: list[float], zoom: float, u: float
) -> tuple[float, float, float, float]:
    """The crop box (x0, y0, x1, y1) a fraction u of the way from the full frame to `zoom`x
    centred on `focus` (kept inside the frame)."""
    w, h = width / zoom, height / zoom
    x0 = min(max(focus[0] - w / 2, 0.0), width - w)
    y0 = min(max(focus[1] - h / 2, 0.0), height - h)
    lerp = lambda a, b: round(a + (b - a) * u, 3)  # noqa: E731
    return (lerp(0.0, x0), lerp(0.0, y0), lerp(width, x0 + w), lerp(height, y0 + h))


def music_points(music: Music, duration: float, duck: tuple[float, float] | None) -> list:
    """(time, linear gain) points: fade in, the level, a duck around the voice, fade out."""
    level = 10 ** (music.level_db / 20)
    pts = [(0.0, 0.0), (min(music.fade_in_s, duration), level)]
    if duck is not None:
        q, a = duck
        pts += [(max(q - music.duck_s, 0.0), level), (q, 0.0), (a, 0.0), (a + music.duck_s, level)]
    pts += [(max(duration - music.fade_out_s, 0.0), level), (duration, 0.0)]
    out: list[tuple[float, float]] = []
    for t, g in pts:  # keep time order; a duck too close to an end wins over the level
        t = round(min(max(t, 0.0), duration), 3)
        if out and t <= out[-1][0]:
            t = out[-1][0]
        out.append((t, round(g, 6)))
    return out


def volume_expr(points: list[tuple[float, float]]) -> str:
    """An ffmpeg `volume` expression (eval=frame) that follows the points linearly."""
    expr = f"{points[-1][1]:g}"
    for (t0, g0), (t1, g1) in reversed(list(pairwise(points))):
        if t1 <= t0:
            continue
        ramp = f"{g0:g}+({g1 - g0:g})*(t-{t0:g})/{t1 - t0:g}"
        expr = f"if(lt(t,{t1:g}),{ramp},{expr})"
    return expr


def build_plan(tl: Timeline, rec: dict) -> Plan:
    """The cut: output frames, overlays, audio cues, captions and the music curve.

    `rec` is events.json: {"shots": {id: {"frames": [t, ...], "events": {name: t},
    "audio": {"question": path, "answer": path}}}}.
    """
    plan = Plan(fps=tl.fps)
    shots = rec.get("shots", {})
    duck = None
    for shot in tl.shots:
        if shot.id not in shots:
            raise TimelineError(f"events.json has no shot {shot.id}")
        r = shots[shot.id]
        times = r["frames"]
        if not times:
            raise TimelineError(f"shot {shot.id}: no frames")
        segs = segments(shot, r["events"])
        start = len(plan.frames) / tl.fps
        total = sum(s.out_s for s in segs)
        n = round(total * tl.fps)
        for k in range(n):
            t = src_time(segs, k / tl.fps)
            plan.frames.append((shot.id, max(bisect.bisect_right(times, t) - 1, 0)))
        end = len(plan.frames) / tl.fps
        plan.spans[shot.id] = (start, end)
        if shot.caption:
            plan.overlays.append(Overlay(overlay_name("caption", shot.id), start, end))
            plan.captions.append((start, end, shot.caption))
        if shot.tag:
            plan.overlays.append(Overlay(overlay_name("tag", shot.id), start, end))
        acc = start
        for s in segs:
            if s.speed != 1:
                plan.sped.append((acc, acc + s.out_s, s.speed))
                plan.overlays.append(
                    Overlay(overlay_name("label", s.speed), acc, min(acc + s.out_s, end))
                )
            acc += s.out_s
        ev = r["events"]
        if shot.zoom > 1:
            if "focus" not in r:
                raise TimelineError(f"shot {shot.id}: events.json lacks the zoom's focus")
            first = round(start * tl.fps)
            begin = out_time(segs, ev["dialog"])
            for k in range(n):
                u = (k / tl.fps - begin) / shot.zoom_ease_s if shot.zoom_ease_s else 1.0
                if u > 0:
                    box = zoom_box(tl.width, tl.height, r["focus"], shot.zoom, ease(min(u, 1)))
                    plan.zooms[first + k] = box
        if shot.id == tl.thumbnail_shot:
            t = src_time(segs, max(out_time(segs, ev["end"]) - 0.1, 0.0))
            plan.thumbnail = (shot.id, max(bisect.bisect_right(times, t) - 1, 0))
        if shot.kind == "voice":
            audio = r.get("audio", {})
            if "question" not in audio or "answer" not in audio:
                raise TimelineError(f"shot {shot.id}: events.json lacks its question/answer audio")
            q = start + out_time(segs, ev["rec_start"])
            a = start + out_time(segs, ev["audio"])
            plan.cues += [Cue(audio["question"], q), Cue(audio["answer"], a)]
            duck = (q, start + out_time(segs, ev["audio_end"]))
    plan.music = music_points(tl.music, plan.duration, duck)
    check_labels(plan)
    return plan


def check_labels(plan: Plan) -> None:
    """Every stretch played above 1x carries the label for its whole length."""
    for a, b, speed in plan.sped:
        name = overlay_name("label", speed)
        covered = [(o.start, o.end) for o in plan.overlays if o.image == name]
        if not any(s <= a + 1e-9 and e >= min(b, plan.duration) - 1e-9 for s, e in covered):
            raise TimelineError(f"the {speed:g}x stretch {a:.2f}-{b:.2f} s has no label")


def srt_time(t: float) -> str:
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt(plan: Plan) -> str:
    return "".join(
        f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n"
        for i, (a, b, text) in enumerate(plan.captions, 1)
    )


def overlay_filter(plan: Plan, first_input: int = 1) -> tuple[list[str], str]:
    """The overlay PNGs (one ffmpeg input each, from `first_input`) and the filtergraph that
    lays them over input 0 when each is due. The graph's output is [v]."""
    images = sorted({o.image for o in plan.overlays})
    index = {img: first_input + i for i, img in enumerate(images)}
    chain, last = [], "0:v"
    for n, o in enumerate(plan.overlays):
        out = f"o{n}"
        chain.append(
            f"[{last}][{index[o.image]}:v]overlay=0:0:"
            f"enable='between(t,{o.start:.4f},{o.end - 1e-4:.4f})'[{out}]"
        )
        last = out
    chain.append(f"[{last}]scale=in_range=full:out_range=tv,format=yuv420p[v]")  # JPEG range
    return images, ";\n".join(chain)


def summary(plan: Plan, tl: Timeline) -> str:
    lines = [
        f"{sid:>10}: {a:6.2f}-{b:6.2f} s ({b - a:5.2f} s)" for sid, (a, b) in plan.spans.items()
    ]
    ok = tl.min_s <= plan.duration <= tl.max_s
    lines.append(
        f"total {plan.duration:.2f} s ({len(plan.frames)} frames); target "
        f"{tl.min_s:g}-{tl.max_s:g} s: {'ok' if ok else 'OUTSIDE'}"
    )
    return "\n".join(lines)


def read_events(folder: Path) -> dict:
    return json.loads((folder / "events.json").read_text())
