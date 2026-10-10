"""Phase 4 real-model tests: TTS → padded, noisy audio → VAD → ASR, plus the latency targets.

Targets (PLAN.md → Voice input pipeline → Latency): end of speech → transcript ≤ 2 s for a
10 s question (VAD + ASR, models loaded), and TTS real-time factor ≤ 0.5.
"""

import re
import time

import mlx.core as mx
import numpy as np
import pytest

from qwn.voice import SAMPLE_RATE, Voice, duration

SENTENCE = "Which supplier delivered the most orders to the northern warehouse last spring"
QUESTION = (
    "Could you tell me which of our suppliers delivered the largest number of orders to the "
    "northern warehouse last spring and what the average delivery time was for those orders"
)
ANSWER = (
    "The northern warehouse received most of its spring orders from a single supplier. "
    "Deliveries took about four days on average, a little faster than the year before.\n"
    "The report also notes two late shipments in April, both caused by bad weather."
)


@pytest.fixture(scope="module")
def voice(registry):
    v = Voice(registry.settings, registry)
    mx.random.seed(0)  # TTS samples: keep the synthetic speech the same from run to run
    return v


@pytest.fixture(scope="module")
def spoken(voice):
    return voice.speak(SENTENCE)


def speech_bounds(audio: np.ndarray) -> tuple[float, float]:
    """First and last 10 ms frame louder than 5% of the peak frame: where the speech is."""
    frame = SAMPLE_RATE // 100
    rms = np.sqrt(np.mean(audio[: audio.size // frame * frame].reshape(-1, frame) ** 2, axis=1))
    loud = np.flatnonzero(rms > 0.05 * rms.max())
    return loud[0] * frame / SAMPLE_RATE, (loud[-1] + 1) * frame / SAMPLE_RATE


def padded(audio: np.ndarray, seconds: float = 1.0, noise: float = 0.003) -> np.ndarray:
    pad = np.zeros(round(seconds * SAMPLE_RATE), dtype=np.float32)
    x = np.concatenate([pad, audio, pad])
    return x + np.random.default_rng(0).normal(0, noise, x.size).astype(np.float32)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = (re.findall(r"[a-z']+", s.lower()) for s in (reference, hypothesis))
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / len(ref)


def test_vad_finds_the_speech_within_150_ms(registry, spoken):
    start, end = speech_bounds(spoken)
    segments = registry.vad().speech_segments(padded(spoken))
    assert len(segments) == 1, segments
    (found_start, found_end) = segments[0]
    assert found_start == pytest.approx(1.0 + start, abs=0.15)
    assert found_end == pytest.approx(1.0 + end, abs=0.15)


def test_asr_on_the_trimmed_speech_matches_the_sentence(voice, spoken):
    transcript = voice.transcribe(padded(spoken))
    assert transcript.chunks == 1
    assert wer(SENTENCE, transcript.text) <= 0.10, transcript.text


def test_silence_and_noise_have_no_speech(registry):
    vad = registry.vad()
    assert vad.speech_segments(np.zeros(3 * SAMPLE_RATE, dtype=np.float32)) == []
    noise = np.random.default_rng(1).normal(0, 0.05, 3 * SAMPLE_RATE).astype(np.float32)
    assert vad.speech_segments(noise) == []


def test_end_of_speech_to_transcript_within_2_s_for_a_10_s_question(voice):
    audio = voice.speak(QUESTION)
    assert 7.0 <= duration(audio) <= 14.0, duration(audio)  # "a 10 s question"
    recording = padded(audio, seconds=0.5)
    voice.transcribe(recording)  # warm-up: first call compiles kernels
    start = time.perf_counter()
    transcript = voice.transcribe(recording)
    elapsed = time.perf_counter() - start
    print(f"\n10 s question: {duration(audio):.1f} s audio → transcript in {elapsed:.2f} s")
    assert elapsed <= 2.0
    assert wer(QUESTION, transcript.text) <= 0.10, transcript.text


def test_tts_real_time_factor_at_most_half(voice):
    voice.speak("Warming up.")
    start = time.perf_counter()
    audio = voice.speak(ANSWER)
    elapsed = time.perf_counter() - start
    rtf = elapsed / duration(audio)
    print(f"\nTTS: {duration(audio):.1f} s audio in {elapsed:.2f} s, RTF {rtf:.2f}")
    assert rtf <= 0.5
