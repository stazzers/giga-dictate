"""Bounded microphone capture, native audio clocks, and streaming resampling."""
import numpy as np

from .meeting_types import AudioFrame
from .process_audio import ProcessInputStream


class MicrophoneStream:
    def __init__(self, emit, fail, device=None):
        def on_pcm(pcm, sequence, rate, timestamp):
            emit(AudioFrame("microphone", sequence, timestamp, rate, pcm))

        self.stream = ProcessInputStream(device=device, samplerate=None, blocksize=None, channels=1,
                                         dtype="float32", timed=True, callback=on_pcm, on_failure=fail)

    def start(self):
        try:
            self.stream.start()
        except RuntimeError:
            raise ValueError("Микрофон не передал звук. Проверьте устройство и разрешение macOS.") from None

    def stop(self):
        self.stream.close()


class FrameNormalizer:
    """Resample once per source, retaining filter state across callbacks.

    Buffer PTS establishes the timeline. Sample counts smooth timestamp jitter;
    discontinuities reset the filter and remain visible as gaps, never compressed.
    """
    def __init__(self, origin):
        self.origin = origin
        self.rate = None
        self.converter = None
        self.next_native = None
        self.next_output = 0
        self.sequence = None

    def convert(self, frame):
        pcm = np.asarray(frame.pcm, dtype=np.float32)
        if pcm.ndim == 2:
            pcm = pcm.mean(axis=1)
        if pcm.ndim != 1 or not np.isfinite(pcm).all() or frame.sample_rate <= 0:
            raise ValueError("Некорректный аудиобуфер")
        discontinuity = (self.next_native is not None and
                         (abs(frame.start - self.next_native) > .05 or
                          frame.sequence != self.sequence + 1))
        if self.rate != frame.sample_rate or discontinuity:
            self.rate = frame.sample_rate
            if self.rate != 16000:
                import soxr
                self.converter = soxr.ResampleStream(self.rate, 16000, 1, dtype="float32")
            else:
                self.converter = None
            self.next_output = round((frame.start - self.origin) * 16000)
        # Account for soxr's retained filter tail before comparing native clocks.
        # Correct slow drift by stretching at most 0.1% of each block, instead
        # of periodically jumping PTS and introducing artificial audio gaps.
        expected = round((frame.start - self.origin) * 16000)
        delay = self.converter.delay() if self.converter else 0
        drift = round(expected - self.next_output - delay)
        out = self.converter.resample_chunk(pcm) if self.converter else pcm
        if len(out) > 1 and abs(drift) >= 8:
            limit = max(1, round(len(out) * .001))
            adjustment = max(-limit, min(limit, drift))
            out = np.interp(np.linspace(0, len(out) - 1, len(out) + adjustment),
                            np.arange(len(out)), out).astype(np.float32)
        start_sample = self.next_output
        self.next_output += len(out)
        self.next_native = frame.start + len(pcm) / self.rate
        self.sequence = frame.sequence
        if start_sample < 0:
            out = out[min(len(out), -start_sample):]
            start_sample = 0
        return start_sample / 16000, np.asarray(out, dtype=np.float32), discontinuity

    def flush(self):
        start = self.next_output / 16000
        if self.converter is None:
            return start, np.empty(0, np.float32)
        out = self.converter.resample_chunk(np.empty(0, np.float32), last=True)
        self.converter = None
        return start, out
