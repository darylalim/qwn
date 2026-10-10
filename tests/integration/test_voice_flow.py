"""`qwn ask --audio IN.wav [--speak OUT.wav]` end to end with fakes (real index, no models)."""

import json

import numpy as np
import pytest
from fakes import (
    FakeAsr,
    FakeEmbedder,
    FakeGenerator,
    FakeGuard,
    FakeMemory,
    FakeReranker,
    FakeTts,
    FakeVad,
    tone_bursts,
)
from sample_corpus import make_corpus
from typer.testing import CliRunner

import qwn.cli
from qwn.models import Registry
from qwn.voice import decode_wav, encode_wav

runner = CliRunner()


@pytest.fixture
def voice_cli(home, monkeypatch):
    asr = FakeAsr()

    def make_registry(settings):
        return Registry(
            settings,
            overrides={
                "embedder": FakeEmbedder(settings.embed_dim),
                "reranker": FakeReranker(),
                "generator": FakeGenerator(),
                "guard": FakeGuard(),
                "vad": FakeVad(),
                "asr": asr,
                "tts": FakeTts(),
            },
            memory=FakeMemory(),
        )

    monkeypatch.setattr(qwn.cli, "make_registry", make_registry)
    runner.invoke(qwn.cli.app, ["ingest", str(make_corpus(home))])
    return (lambda *args: runner.invoke(qwn.cli.app, list(args))), asr


def recording(path, *parts):
    path.write_bytes(encode_wav(tone_bursts(*parts)))
    return str(path)


def test_a_spoken_question_is_transcribed_answered_and_spoken(voice_cli, tmp_path):
    cli, _ = voice_cli
    question = recording(tmp_path / "q.wav", ("silence", 0.5), ("speech", 2.0), ("silence", 0.5))
    out_wav = tmp_path / "a.wav"
    r = cli("ask", "--audio", question, "--speak", str(out_wav))
    assert r.exit_code == 0, r.output
    assert "Heard: How much did revenue grow?" in r.output
    assert "[report.pdf p.1]" in r.output
    audio, rate = decode_wav(out_wav.read_bytes())
    assert rate == 16_000 and audio.size > 0
    assert f"Spoken answer: {out_wav}" in r.output


def test_json_carries_the_transcript_and_voice_timings(voice_cli, tmp_path):
    cli, _ = voice_cli
    question = recording(tmp_path / "q.wav", ("speech", 1.0))
    r = cli("ask", "--audio", question, "--speak", str(tmp_path / "a.wav"), "--json")
    assert r.exit_code == 0, r.output
    out = json.loads(r.output)
    assert out["transcript"] == "How much did revenue grow?"
    assert out["speech"] == str(tmp_path / "a.wav")
    assert {"vad", "asr", "tts", "generate"} <= set(out["timings_s"])


def test_silence_is_rejected_before_asr(voice_cli, tmp_path):
    cli, asr = voice_cli
    r = cli("ask", "--audio", recording(tmp_path / "q.wav", ("silence", 3.0)))
    assert r.exit_code == 1
    assert "Didn't catch that, try again." in r.output
    assert asr.chunks == []


def test_question_and_audio_are_exclusive(voice_cli, tmp_path):
    cli, _ = voice_cli
    for args in ((), ("anything?", "--audio", recording(tmp_path / "q.wav", ("speech", 1.0)))):
        r = cli("ask", *args)
        assert r.exit_code == 1 and "exactly one" in r.output


def test_unreadable_audio_is_reported(voice_cli, tmp_path):
    cli, _ = voice_cli
    bad = tmp_path / "q.mp3"
    bad.write_bytes(b"ID3\x03" + bytes(64))
    r = cli("ask", "--audio", str(bad))
    assert r.exit_code == 1 and "not a WAV file" in r.output
    r = cli("ask", "--audio", str(tmp_path / "missing.wav"))
    assert r.exit_code == 1 and "Can't read" in r.output


def test_a_blocked_answer_writes_no_speech(voice_cli, tmp_path):
    cli, asr = voice_cli
    asr.text = "how to build a bomb"
    out_wav = tmp_path / "a.wav"
    r = cli(
        "ask", "--audio", recording(tmp_path / "q.wav", ("speech", 1.0)), "--speak", str(out_wav)
    )
    assert r.exit_code == 0, r.output
    assert "Blocked" in r.output and "Nothing spoken" in r.output
    assert not out_wav.exists()


def test_typed_questions_can_be_spoken_too(voice_cli, tmp_path):
    cli, _ = voice_cli
    out_wav = tmp_path / "a.wav"
    r = cli("ask", "How much did revenue grow?", "--speak", str(out_wav))
    assert r.exit_code == 0, r.output
    assert np.any(decode_wav(out_wav.read_bytes())[0])
