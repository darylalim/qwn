"""Voice: recording → VAD → ASR transcript; answer text → speech (PLAN.md → Voice input pipeline).

Audio is mono float32 at 16 kHz everywhere inside qwn. The pure logic (WAV decoding, resampling,
silence rejection, trimming, splitting at pauses) lives here and is unit-tested with a fake `Vad`;
the models sit behind `qwn.models.Registry`. Transcripts are never logged: only lengths and timings.
"""

import logging
import re
import struct
import time
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from qwn.answer import Response, strip_citations
from qwn.config import Settings
from qwn.models import Registry

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
Audio = NDArray[np.float32]
Span = tuple[float, float]  # (start_s, end_s)

NO_SPEECH = "Didn't catch that, try again."


class AudioError(ValueError):
    """The recording couldn't be read (not a WAV file we understand)."""


class NoSpeech(Exception):
    """VAD found less than `vad_min_speech_ms` of speech: nothing is transcribed."""

    def __init__(self, speech_s: float) -> None:
        super().__init__(NO_SPEECH)
        self.speech_s = speech_s


# decoding and encoding


_PCM, _FLOAT, _EXTENSIBLE = 1, 3, 0xFFFE


def decode_wav(data: bytes) -> tuple[Audio, int]:
    """(mono float32 in [-1, 1], sample rate) from a RIFF/WAVE file: PCM 8/16/24/32-bit or
    IEEE float 32/64-bit. Channels are averaged."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("not a WAV file")
    fmt: tuple[int, int, int, int] | None = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if chunk_id == b"fmt ":
            if len(body) < 16:
                raise AudioError("broken WAV header")
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == _EXTENSIBLE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]  # first two bytes of the sub-format GUID
            fmt = (tag, channels, rate, bits)
        elif chunk_id == b"data":
            if fmt is None:
                raise AudioError("WAV data before its format")
            return _samples(body, *fmt), fmt[2]
        pos += 8 + size + (size & 1)  # chunks are padded to an even length
    raise AudioError("WAV file has no audio data")


def _samples(body: bytes, tag: int, channels: int, rate: int, bits: int) -> Audio:
    if channels < 1 or rate < 1:
        raise AudioError("broken WAV header")
    width = bits // 8
    usable = len(body) - len(body) % (width * channels) if width else 0
    raw = body[:usable]
    if tag == _PCM and bits == 8:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif tag == _PCM and bits == 16:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif tag == _PCM and bits == 24:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float32) / (1 << 23)
    elif tag == _PCM and bits == 32:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2**31
    elif tag == _FLOAT and bits in (32, 64):
        x = np.frombuffer(raw, "<f4" if bits == 32 else "<f8").astype(np.float32)
    else:
        raise AudioError(f"unsupported WAV encoding (format {tag}, {bits}-bit)")
    x = x.reshape(-1, channels).mean(axis=1)
    return np.clip(np.nan_to_num(x), -1.0, 1.0).astype(np.float32)


def resample(audio: Audio, rate_from: int, rate_to: int) -> Audio:
    """Band-limited resampling in the frequency domain (fine for recordings of a few minutes)."""
    if rate_from == rate_to or audio.size == 0:
        return audio.astype(np.float32)
    n_out = round(audio.size * rate_to / rate_from)
    spectrum = np.fft.rfft(audio.astype(np.float64))
    bins = n_out // 2 + 1
    if bins <= spectrum.size:
        spectrum = spectrum[:bins]
    else:
        spectrum = np.pad(spectrum, (0, bins - spectrum.size))
    out = np.fft.irfft(spectrum, n_out) * (n_out / audio.size)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def load_audio(source: bytes | Path) -> Audio:
    """A WAV recording (bytes, or a file) as mono float32 at 16 kHz."""
    data = source.read_bytes() if isinstance(source, Path) else source
    audio, rate = decode_wav(data)
    return resample(audio, rate, SAMPLE_RATE)


def encode_wav(audio: Audio, rate: int = SAMPLE_RATE) -> bytes:
    """16-bit PCM mono WAV (what `--speak` writes and the Chat page plays)."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        _PCM,
        1,
        rate,
        rate * 2,
        2,
        16,
        b"data",
        len(pcm),
    )
    return header + pcm


def duration(audio: Audio) -> float:
    return audio.size / SAMPLE_RATE


# trimming and splitting


def plan_chunks(segments: list[Span], length_s: float, settings: Settings) -> list[Span]:
    """The spans to transcribe: speech trimmed to its first..last segment ± `vad_pad_ms`
    (clamped to the recording), split at the longest pauses into chunks ≤ `asr_max_seconds`.

    Raises NoSpeech if the segments add up to less than `vad_min_speech_ms`.
    """
    speech_s = sum(end - start for start, end in segments)
    if not segments or speech_s * 1000 < settings.vad_min_speech_ms:
        raise NoSpeech(speech_s)
    pad = settings.vad_pad_ms / 1000
    span = (max(0.0, segments[0][0] - pad), min(length_s, segments[-1][1] + pad))
    return split_at_pauses(segments, span, float(settings.asr_max_seconds))


def split_at_pauses(segments: list[Span], span: Span, max_s: float) -> list[Span]:
    """Cut `span` in the middle of its longest pause, recursively, until every piece is
    ≤ `max_s`. A piece with no pause left (one long run of speech) is cut every `max_s`."""
    start, end = span
    if end - start <= max_s:
        return [span]
    pauses = [
        (b[0] - a[1], (a[1] + b[0]) / 2)
        for a, b in pairwise(segments)
        if start < (a[1] + b[0]) / 2 < end
    ]
    if not pauses:
        cuts = [*np.arange(start, end, max_s).tolist(), end]
        return [(a, b) for a, b in pairwise(cuts) if b > a]
    _, cut = max(pauses)
    return split_at_pauses(segments, (start, cut), max_s) + split_at_pauses(
        segments, (cut, end), max_s
    )


def clip(audio: Audio, span: Span) -> Audio:
    return audio[round(span[0] * SAMPLE_RATE) : round(span[1] * SAMPLE_RATE)]


# what gets spoken

_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_EMPHASIS = re.compile(r"(\*\*|__|\*|`)")
_LINE_MARK = re.compile(r"^\s*(#{1,6}\s+|[-*+]\s+|>\s*)", re.MULTILINE)


def speakable(text: str) -> str:
    """Answer text for TTS: no `[S#]` markers, no markdown symbols, links read as their text."""
    text = strip_citations(text)
    text = _LINK.sub(r"\1", text)
    text = _LINE_MARK.sub("", text)
    text = _EMPHASIS.sub("", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def spoken_text(response: Response) -> str | None:
    """What to read aloud for an answer; None for an answer Guard blocked (nothing to say)."""
    if response.blocked is not None:
        return None
    return speakable(response.text) or None


# the pipeline


@dataclass(frozen=True)
class Transcript:
    text: str
    speech_s: float  # total speech VAD found
    chunks: int  # pieces sent to ASR
    timings: dict[str, float] = field(default_factory=dict)  # seconds: vad, asr


class Voice:
    """Transcribe recordings and speak answers through the registry's voice models."""

    def __init__(self, settings: Settings, registry: Registry) -> None:
        self.settings = settings
        self.registry = registry

    def transcribe(self, audio: Audio) -> Transcript:
        """VAD, then ASR on the trimmed speech. Raises NoSpeech before ASR loads or runs."""
        timings: dict[str, float] = {}
        start = time.perf_counter()
        segments = self.registry.vad().speech_segments(audio)
        timings["vad"] = time.perf_counter() - start
        speech_s = sum(e - s for s, e in segments)
        try:
            chunks = plan_chunks(segments, duration(audio), self.settings)
        except NoSpeech:
            logger.info(
                "no speech: audio_s=%.2f speech_s=%.2f vad_s=%.3f",
                duration(audio),
                speech_s,
                timings["vad"],
            )
            raise
        start = time.perf_counter()
        asr = self.registry.asr()
        texts = [asr.transcribe(clip(audio, span)).strip() for span in chunks]
        timings["asr"] = time.perf_counter() - start
        text = " ".join(t for t in texts if t)
        logger.info(
            "transcribed: audio_s=%.2f speech_s=%.2f chunks=%d transcript_len=%d timings=%s",
            duration(audio),
            speech_s,
            len(chunks),
            len(text),
            {k: round(v, 3) for k, v in timings.items()},
        )
        return Transcript(text=text, speech_s=speech_s, chunks=len(chunks), timings=timings)

    def speak(self, text: str) -> Audio:
        start = time.perf_counter()
        audio = self.registry.tts().synthesize(text)
        elapsed = time.perf_counter() - start
        logger.info(
            "spoke: text_len=%d audio_s=%.2f tts_s=%.2f rtf=%.2f",
            len(text),
            duration(audio),
            elapsed,
            elapsed / max(duration(audio), 1e-6),
        )
        return audio
