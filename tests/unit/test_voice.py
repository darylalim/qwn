"""qwn.voice: WAV decoding, resampling, trimming and splitting, and the pipeline with fakes."""

import logging
import struct

import numpy as np
import pytest
from fakes import FakeAsr, FakeTts, FakeVad, tone_bursts

from qwn.answer import Response
from qwn.config import Settings
from qwn.guard import Check
from qwn.interfaces import Verdict
from qwn.models import Registry
from qwn.voice import (
    SAMPLE_RATE,
    AudioError,
    NoSpeech,
    Voice,
    decode_wav,
    encode_wav,
    load_audio,
    plan_chunks,
    resample,
    speakable,
    split_at_pauses,
    spoken_text,
)


def wav(samples: np.ndarray, rate: int, *, channels: int = 1, bits: int = 16, tag: int = 1):
    """A WAV file with interleaved `samples` (already in the target integer/float dtype)."""
    data = samples.tobytes()
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * channels * bits // 8, 4, bits)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"LIST" + struct.pack("<I", 3) + b"abc\x00"  # odd-sized chunk, padded
    chunks += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


# decoding


def test_pcm16_stereo_is_averaged_to_mono():
    left = np.array([16384, -16384, 0], dtype="<i2")
    right = np.array([0, 0, 32767], dtype="<i2")
    stereo = np.stack([left, right], axis=1).reshape(-1)
    audio, rate = decode_wav(wav(stereo, 44_100, channels=2))
    assert rate == 44_100
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, [0.25, -0.25, 0.5], atol=1e-4)


@pytest.mark.parametrize(
    ("samples", "bits", "tag"),
    [
        (np.array([192, 64], dtype=np.uint8), 8, 1),
        (np.array([2**30, -(2**30)], dtype="<i4"), 32, 1),
        (np.array([0.5, -0.5], dtype="<f4"), 32, 3),
        (np.array([0.5, -0.5], dtype="<f8"), 64, 3),
    ],
)
def test_other_encodings(samples, bits, tag):
    audio, _ = decode_wav(wav(samples, 16_000, bits=bits, tag=tag))
    np.testing.assert_allclose(audio, [0.5, -0.5], atol=1e-2)


def test_pcm24():
    def le24(v: int) -> bytes:
        return (v & 0xFFFFFF).to_bytes(3, "little")

    raw = np.frombuffer(le24(2**22) + le24(-(2**22)), dtype=np.uint8)
    audio, _ = decode_wav(wav(raw, 16_000, bits=24))
    np.testing.assert_allclose(audio, [0.5, -0.5], atol=1e-4)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "not a WAV"),
        (b"ID3\x03" + b"\x00" * 40, "not a WAV"),
        (b"RIFF\x04\x00\x00\x00WAVE", "no audio data"),
        (wav(np.zeros(4, "<i2"), 16_000, tag=0x55), "unsupported WAV encoding"),
    ],
)
def test_unreadable_audio_is_an_error(data, message):
    with pytest.raises(AudioError, match=message):
        decode_wav(data)


def test_encode_then_load_round_trips_at_16k(tmp_path):
    tone = tone_bursts(("speech", 0.5))
    path = tmp_path / "q.wav"
    path.write_bytes(encode_wav(tone))
    back = load_audio(path)
    assert back.size == tone.size
    np.testing.assert_allclose(back, tone, atol=1e-3)


# resampling


@pytest.mark.parametrize("rate", [8_000, 24_000, 44_100, 48_000])
def test_resampling_keeps_length_pitch_and_level(rate):
    t = np.arange(rate) / rate  # 1 s
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out = resample(tone, rate, SAMPLE_RATE)
    assert out.size == SAMPLE_RATE
    peak_hz = np.argmax(np.abs(np.fft.rfft(out))) * SAMPLE_RATE / out.size
    assert peak_hz == pytest.approx(440, abs=2)
    assert np.sqrt(np.mean(out**2)) == pytest.approx(0.5 / np.sqrt(2), rel=0.02)


def test_resampling_drops_what_16k_cant_hold():
    rate = 48_000
    t = np.arange(rate) / rate
    high = (0.5 * np.sin(2 * np.pi * 12_000 * t)).astype(np.float32)  # above 8 kHz Nyquist
    assert np.max(np.abs(resample(high, rate, SAMPLE_RATE))) < 0.01


# trimming and splitting


def test_too_little_speech_is_rejected(home):
    with pytest.raises(NoSpeech, match="Didn't catch that"):
        plan_chunks([], 5.0, Settings())
    with pytest.raises(NoSpeech):
        plan_chunks([(1.0, 1.1), (2.0, 2.1)], 5.0, Settings(vad_min_speech_ms=250))  # 200 ms


def test_trim_pads_and_clamps_to_the_recording(home):
    settings = Settings(vad_pad_ms=200)
    assert plan_chunks([(1.0, 2.0), (2.5, 3.0)], 5.0, settings) == [(0.8, 3.2)]
    assert plan_chunks([(0.1, 4.95)], 5.0, settings) == [(0.0, 5.0)]


def test_long_recordings_split_at_the_longest_pauses(home):
    # pauses: 2 s after 40 s, 5 s after 95 s, 1 s after 150 s
    segments = [(0.0, 40.0), (42.0, 95.0), (100.0, 150.0), (151.0, 200.0)]
    chunks = plan_chunks(segments, 200.0, Settings(asr_max_seconds=120, vad_pad_ms=0))
    assert chunks == [(0.0, 97.5), (97.5, 200.0)]  # cut in the middle of the 5 s pause
    chunks = plan_chunks(segments, 200.0, Settings(asr_max_seconds=60, vad_pad_ms=0))
    assert chunks == [(0.0, 41.0), (41.0, 97.5), (97.5, 150.5), (150.5, 200.0)]
    assert all(b - a <= 60 for a, b in chunks)


def test_speech_with_no_pause_is_cut_every_max_seconds():
    assert split_at_pauses([(0.0, 25.0)], (0.0, 25.0), 10.0) == [
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 25.0),
    ]


# what gets spoken


def test_speakable_drops_citations_and_markdown():
    text = "## Revenue\n\n- **Q3** grew 12% [S1].\n- See [the report](http://x) [S2]\n"
    assert speakable(text) == "Revenue\nQ3 grew 12%.\nSee the report"


def _response(text: str, checks: list[Check] | None = None) -> Response:
    return Response(text, [], [], [], [], False, 0, 0, checks=checks or [])


def test_blocked_answers_are_not_spoken():
    block = Check("response", Verdict("Unsafe", ["Violent"]), "block")
    assert spoken_text(_response("Blocked: …", [block])) is None
    assert spoken_text(_response("Revenue grew [S1].")) == "Revenue grew."


# the pipeline


@pytest.fixture
def voice(home):
    settings = Settings()
    vad, asr, tts = FakeVad(), FakeAsr(), FakeTts()
    registry = Registry(settings, overrides={"vad": vad, "asr": asr, "tts": tts})
    return Voice(settings, registry), vad, asr, tts


def test_transcribe_trims_silence_before_asr(voice):
    v, _, asr, _ = voice
    audio = tone_bursts(("silence", 2.0), ("speech", 1.0), ("silence", 0.2), ("speech", 0.5))
    audio = np.concatenate([audio, tone_bursts(("silence", 3.0))])
    transcript = v.transcribe(audio)
    assert transcript.text == "How much did revenue grow?"
    assert transcript.chunks == 1
    assert transcript.speech_s == pytest.approx(1.7, abs=0.07)
    assert asr.chunks == [pytest.approx(1.7 + 0.4, abs=0.07)]  # speech + 200 ms each side
    assert set(transcript.timings) == {"vad", "asr"}


def test_silent_or_quiet_recordings_never_reach_asr(voice):
    v, vad, asr, _ = voice
    rng = np.random.default_rng(0)
    for audio in (
        tone_bursts(("silence", 3.0)),
        rng.normal(0, 0.01, 3 * SAMPLE_RATE).astype(np.float32),  # low noise
        tone_bursts(("silence", 1.0), ("speech", 0.1), ("silence", 1.0)),  # a click
        np.zeros(0, dtype=np.float32),
    ):
        with pytest.raises(NoSpeech):
            v.transcribe(audio)
    assert vad.calls == 4
    assert asr.chunks == []


def test_long_questions_go_to_asr_in_chunks(home):
    settings = Settings(asr_max_seconds=3)
    asr = FakeAsr("part")
    registry = Registry(settings, overrides={"vad": FakeVad(), "asr": asr})
    audio = tone_bursts(("speech", 2.0), ("silence", 1.0), ("speech", 2.0))
    transcript = Voice(settings, registry).transcribe(audio)
    assert transcript.chunks == 2
    assert transcript.text == "part part"
    assert all(c <= 3.0 for c in asr.chunks)


def test_transcripts_are_never_logged(voice, caplog):
    v, _, asr, tts = voice
    asr.text = "SENTINEL-TRANSCRIPT"
    with caplog.at_level(logging.DEBUG):
        v.transcribe(tone_bursts(("speech", 1.0)))
        v.speak("SENTINEL-ANSWER")
    assert "SENTINEL" not in caplog.text
    assert "transcript_len=19" in caplog.text
    assert tts.texts == ["SENTINEL-ANSWER"]
