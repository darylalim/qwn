"""The demo video's timeline (scripts/demo_video/timeline.py): stdlib only, no ffmpeg or Chrome."""

import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
TIMELINE = ROOT / "docs" / "video" / "timeline.toml"


def _load():
    path = ROOT / "scripts" / "demo_video" / "timeline.py"
    spec = importlib.util.spec_from_file_location("demo_timeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tl_mod = _load()
TimelineError = tl_mod.TimelineError

MINI = """
[video]
fps = 10
min_s = 1
max_s = 60
[thumbnail]
shot = "a"
[[shot]]
id = "a"
kind = "chat"
query = "hard-033"
caption = "Ask"
waits = [{ from = "sent", to = "answer", speed = 4 }]
"""


def _queries(name: str) -> dict[str, dict]:
    lines = (ROOT / "eval" / name / "queries.jsonl").read_text().splitlines()
    return {(q := json.loads(line))["id"]: q for line in lines if line.strip()}


def _rec(shot_events: dict[str, dict[str, float]], fps: float = 20.0) -> dict:
    """A fake events.json: frames every 1/fps s up to each shot's end."""
    shots = {}
    for sid, ev in shot_events.items():
        n = int(ev["end"] * fps) + 1
        shots[sid] = {"frames": [i / fps for i in range(n)], "events": ev}
    return {"shots": shots}


def test_committed_timeline_parses_and_names_real_queries():
    tl = tl_mod.load(TIMELINE)
    hard, turns = _queries("hard"), _queries("turns")
    kinds = [s.kind for s in tl.shots]
    assert kinds[-1] == "card"
    for s in tl.shots:
        if s.kind == "followup":
            q = turns[s.query]
            assert len(q["history"]) == 1 and q["expected"]
        elif s.kind == "refusal":
            assert not hard[s.query]["expected"]  # unanswerable
        elif s.query is not None:
            assert hard[s.query]["expected"] and hard[s.query]["answer_contains"]
    assert tl.thumbnail_shot == tl.shots[0].id
    # every caption, tag and speed label has an overlay PNG to render
    names = tl_mod.overlays(tl)
    assert sum(style == "caption" for style, _ in names.values()) == len(tl.shots) - 1
    assert ("label", "Sped up 10\N{MULTIPLICATION SIGN}") in names.values()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (('kind = "chat"', 'kind = "dance"'), "kind must be one of"),
        (('caption = "Ask"\n', ""), "needs a caption"),
        (("speed = 4", "speed = 1"), "above 1"),
        (('from = "sent", to = "answer"', 'from = "answer", to = "sent"'), "backwards"),
        (('from = "sent"', 'from = "typed"'), "events are"),
        (('query = "hard-033"\n', ""), "needs query"),
        (('shot = "a"', 'shot = "b"'), "thumbnail shot"),
        (("[[shot]]", "[[shot]]\nwobble = 1"), "unknown keys"),
    ],
)
def test_parse_rejects(change, message):
    with pytest.raises(TimelineError, match=message):
        tl_mod.parse(MINI.replace(*change))


def test_parse_rejects_overlapping_waits_and_sped_up_sound():
    text = MINI.replace(
        'waits = [{ from = "sent", to = "answer", speed = 4 }]',
        'waits = [{ from = "start", to = "answer", speed = 4 },'
        ' { from = "sent", to = "end", speed = 4 }]',
    )
    with pytest.raises(TimelineError, match="overlap"):
        tl_mod.parse(text)
    voice = MINI.replace('kind = "chat"', 'kind = "voice"').replace(
        'from = "sent", to = "answer"', 'from = "rec_start", to = "answer"'
    )
    with pytest.raises(TimelineError, match="must play at 1x"):
        tl_mod.parse(voice)


def test_open_page_must_follow_a_chat_shot():
    text = MINI.replace('kind = "chat"', 'kind = "open_page"').replace('query = "hard-033"\n', "")
    text = text.replace('from = "sent", to = "answer"', 'from = "click", to = "dialog"')
    with pytest.raises(TimelineError, match="must follow a chat shot"):
        tl_mod.parse(text)


def test_segments_split_at_waits_and_times_invert():
    shot = tl_mod.parse(MINI).shots[0]
    ev = {"start": 0.0, "sent": 2.0, "answer": 22.0, "end": 25.0}
    segs = tl_mod.segments(shot, ev)
    assert [(s.t0, s.t1, s.speed) for s in segs] == [(0, 2, 1), (2, 22, 4), (22, 25, 1)]
    assert sum(s.out_s for s in segs) == pytest.approx(2 + 5 + 3)
    for t in (0.0, 1.0, 2.0, 10.0, 22.0, 24.0):
        assert tl_mod.src_time(segs, tl_mod.out_time(segs, t)) == pytest.approx(t)
    assert tl_mod.out_time(segs, 12.0) == pytest.approx(2 + 10 / 4)


def test_segments_need_every_event_in_order():
    shot = tl_mod.parse(MINI).shots[0]
    with pytest.raises(TimelineError, match="lacks"):
        tl_mod.segments(shot, {"start": 0.0, "sent": 1.0, "end": 2.0})
    with pytest.raises(TimelineError, match="out of order"):
        tl_mod.segments(shot, {"start": 0.0, "sent": 3.0, "answer": 2.0, "end": 4.0})


def test_plan_frames_overlays_and_labels():
    tl = tl_mod.parse(MINI)
    rec = _rec({"a": {"start": 0.0, "sent": 2.0, "answer": 22.0, "end": 25.0}})
    plan = tl_mod.build_plan(tl, rec)
    assert len(plan.frames) == 100  # 10 s at 10 fps: 2 + 20/4 + 3
    assert plan.duration == pytest.approx(10.0)
    # 1x before the wait: frame k shows the source frame at k/10 s (source runs at 20 fps)
    assert plan.frames[5] == ("a", 10)
    # sped up 4x: 0.1 s of output = 0.4 s of source = 8 source frames
    assert plan.frames[21][1] - plan.frames[20][1] == 8
    assert plan.sped == [(2.0, 7.0, 4.0)]
    label = next(o for o in plan.overlays if "label" in o.image)
    assert (label.start, label.end) == (2.0, 7.0)
    assert plan.captions == [(0.0, 10.0, "Ask")]
    assert plan.thumbnail[0] == "a"
    assert tl_mod.build_plan(tl, rec) == plan  # deterministic


def test_check_labels_catches_an_unlabelled_stretch():
    tl = tl_mod.parse(MINI)
    rec = _rec({"a": {"start": 0.0, "sent": 2.0, "answer": 22.0, "end": 25.0}})
    plan = tl_mod.build_plan(tl, rec)
    plan.overlays = [o for o in plan.overlays if "label" not in o.image]
    with pytest.raises(TimelineError, match="has no label"):
        tl_mod.check_labels(plan)
    short = tl_mod.build_plan(tl, rec)
    short.overlays = [
        dataclasses.replace(o, end=o.end - 1) if "label" in o.image else o for o in short.overlays
    ]
    with pytest.raises(TimelineError, match="has no label"):
        tl_mod.check_labels(short)


def _voice_rec():
    ev = {
        "start": 0.0,
        "rec_start": 5.0,
        "rec_stop": 11.0,
        "answer": 35.0,
        "audio": 43.0,
        "audio_end": 46.0,
        "end": 50.0,
    }
    rec = _rec({"v": ev})
    rec["shots"]["v"]["audio"] = {"question": "question.wav", "answer": "answer.wav"}
    return rec


VOICE = MINI.replace('shot = "a"', 'shot = "v"').replace(
    """id = "a"
kind = "chat"
query = "hard-033"
caption = "Ask"
waits = [{ from = "sent", to = "answer", speed = 4 }]""",
    """id = "v"
kind = "voice"
query = "hard-022"
caption = "Ask out loud"
tag = "Question voice: synthetic"
waits = [
  { from = "rec_stop", to = "answer", speed = 8 },
  { from = "answer", to = "audio", speed = 8 },
]""",
)


def test_voice_cues_and_music_duck():
    tl = tl_mod.parse(VOICE)
    plan = tl_mod.build_plan(tl, _voice_rec())
    # 5 + 6 at 1x, 24/8 + 8/8 sped, 3 + 4 at 1x
    assert plan.duration == pytest.approx(22.0)
    assert [(c.audio, c.start) for c in plan.cues] == [("question.wav", 5.0), ("answer.wav", 15.0)]
    gains = dict(plan.music)
    assert gains[5.0] == 0.0 and gains[18.0] == 0.0  # silent from the question to the answer's end
    level = 10 ** (-18 / 20)
    assert gains[4.5] == pytest.approx(level, abs=1e-6)  # ducks over 0.5 s before
    assert gains[18.5] == pytest.approx(level, abs=1e-6)
    assert plan.music[0] == (0.0, 0.0) and plan.music[-1] == (22.0, 0.0)
    assert [t for t, _ in plan.music] == sorted(t for t, _ in plan.music)
    assert any("tag-v" in o.image for o in plan.overlays)


def test_music_points_stay_in_time_order_when_a_duck_meets_a_fade():
    music = tl_mod.Music(level_db=-18.0, fade_in_s=1.0, duck_s=0.5, fade_out_s=2.0)
    pts = tl_mod.music_points(music, 10.0, (0.8, 9.5))
    times = [t for t, _ in pts]
    assert times == sorted(times) and times[0] == 0.0 and times[-1] == 10.0
    assert dict(pts)[10.0] == 0.0


def test_voice_needs_its_audio():
    rec = _voice_rec()
    del rec["shots"]["v"]["audio"]
    with pytest.raises(TimelineError, match="audio"):
        tl_mod.build_plan(tl_mod.parse(VOICE), rec)


def _eval_expr(expr: str, t: float) -> float:
    """Evaluate the ffmpeg expression subset volume_expr writes (if, lt, arithmetic)."""
    env = {"if_": lambda c, a, b: a if c else b, "lt": lambda a, b: a < b, "t": t}
    return eval(expr.replace("if(", "if_("), {"__builtins__": {}}, env)


def test_volume_expr_follows_the_points():
    pts = [(0.0, 0.0), (1.0, 0.5), (4.0, 0.5), (4.5, 0.0), (6.0, 0.0), (6.5, 0.5), (9.0, 0.0)]
    expr = tl_mod.volume_expr(pts)
    for t, want in [(0.0, 0.0), (0.5, 0.25), (2.0, 0.5), (4.25, 0.25), (5.0, 0.0), (6.5, 0.5)]:
        assert _eval_expr(expr, t) == pytest.approx(want)
    assert _eval_expr(expr, 20.0) == 0.0


def test_srt_is_stable():
    tl = tl_mod.parse(MINI)
    rec = _rec({"a": {"start": 0.0, "sent": 2.0, "answer": 22.0, "end": 25.0}})
    text = tl_mod.srt(tl_mod.build_plan(tl, rec))
    assert text == "1\n00:00:00,000 --> 00:00:10,000\nAsk\n\n"
    assert tl_mod.srt_time(3723.4567) == "01:02:03,457"


def test_overlay_filter_lists_each_image_once():
    tl = tl_mod.parse(MINI)
    plan = tl_mod.build_plan(
        tl, _rec({"a": {"start": 0.0, "sent": 2.0, "answer": 22.0, "end": 25.0}})
    )
    images, graph = tl_mod.overlay_filter(plan)
    assert images == sorted(set(images)) and len(images) == 2
    assert graph.count("overlay=") == len(plan.overlays)
    assert graph.endswith("out_range=tv,format=yuv420p[v]")


ZOOM = """
[video]
fps = 10
[thumbnail]
shot = "a"
[[shot]]
id = "a"
kind = "chat"
query = "hard-033"
caption = "Ask"
[[shot]]
id = "p"
kind = "open_page"
caption = "Check"
waits = [{ from = "click", to = "dialog", speed = 4 }]
zoom = 2.0
zoom_ease_s = 1.0
"""


def _zoom_rec(**extra):
    rec = _rec(
        {
            "a": {"start": 0.0, "sent": 1.0, "answer": 2.0, "end": 3.0},
            "p": {"start": 0.0, "click": 1.0, "dialog": 5.0, "end": 9.0},
        }
    )
    rec["shots"]["p"].update(extra)
    return rec


def test_zoom_eases_in_on_the_focus_after_the_dialog():
    tl = tl_mod.parse(ZOOM)
    plan = tl_mod.build_plan(tl, _zoom_rec(focus=[1600.0, 1000.0]))
    start = round(plan.spans["p"][0] * 10)
    dialog = start + 20  # 1 s at 1x, then 4 s at 4x
    assert min(plan.zooms) == dialog + 1  # nothing before the dialog is ready
    assert max(plan.zooms) == round(plan.spans["p"][1] * 10) - 1
    # fully in after the ease: 960x540, pushed back inside the frame from the focus
    assert plan.zooms[dialog + 10] == (960.0, 540.0, 1920.0, 1080.0)
    half = plan.zooms[dialog + 5]
    assert half == (480.0, 270.0, 1920.0, 1080.0)  # smoothstep(0.5) = 0.5
    widths = [plan.zooms[k][2] - plan.zooms[k][0] for k in sorted(plan.zooms)]
    assert widths == sorted(widths, reverse=True)  # only ever zooms in


def test_zoom_box_centres_and_clamps():
    assert tl_mod.zoom_box(1920, 1080, [960.0, 540.0], 2.0, 1.0) == (480.0, 270.0, 1440.0, 810.0)
    assert tl_mod.zoom_box(1920, 1080, [0.0, 0.0], 2.0, 1.0) == (0.0, 0.0, 960.0, 540.0)
    assert tl_mod.zoom_box(1920, 1080, [500.0, 500.0], 3.0, 0.0) == (0.0, 0.0, 1920.0, 1080.0)


def test_zoom_needs_a_focus_and_an_open_page_shot():
    with pytest.raises(TimelineError, match="focus"):
        tl_mod.build_plan(tl_mod.parse(ZOOM), _zoom_rec())
    with pytest.raises(TimelineError, match="only an open_page shot zooms"):
        tl_mod.parse(MINI.replace('caption = "Ask"', 'caption = "Ask"\nzoom = 2.0'))
