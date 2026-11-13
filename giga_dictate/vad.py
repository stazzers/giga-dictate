"""Stateful Silero ONNX VAD with bounded phrase buffers."""
from collections import deque

import numpy as np

from .meeting_types import SpeechSegment


class SileroProbability:
    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options,
                                            providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, 64), np.float32)

    def __call__(self, pcm):
        x = np.concatenate((self.context, pcm.reshape(1, -1)), axis=1)
        output, self.state = self.session.run(None, {
            "input": x, "state": self.state, "sr": np.array(16000, dtype=np.int64),
        })
        self.context = x[:, -64:].copy()
        return float(output[0, 0])


class StreamingVAD:
    block_size = 512

    def __init__(self, source, probability, threshold=.5, silence=.5,
                 preroll=.2, max_duration=15.0):
        self.source, self.probability = source, probability
        self.threshold = threshold
        self.silence_blocks = int(np.ceil(silence * 16000 / 512))
        self.max_blocks = int(max_duration * 16000 / 512)
        self.history = deque(maxlen=int(np.ceil(preroll * 16000 / 512)))
        self.parts = []
        self.quiet = 0
        self.start = 0.0
        self.pending = np.empty(0, np.float32)
        self.pending_start = 0.0
        self.expected = None

    def feed(self, start, pcm):
        results = []
        if self.expected is not None and abs(start - self.expected) > .05:
            results.extend(self.flush())
            self.probability.reset()
        self.expected = start + len(pcm) / 16000
        if not self.pending.size:
            self.pending_start = start
        self.pending = np.concatenate((self.pending, pcm))
        offset = 0
        while len(self.pending) - offset >= self.block_size:
            block = self.pending[offset:offset + self.block_size].copy()
            at = self.pending_start + offset / 16000
            results.extend(self._block(at, block))
            offset += self.block_size
        self.pending = self.pending[offset:].copy()
        self.pending_start += offset / 16000
        return results

    def _block(self, at, block):
        speaking = self.probability(block) >= self.threshold
        if not self.parts:
            if not speaking:
                self.history.append((at, block))
                return []
            self.start = self.history[0][0] if self.history else at
            self.parts = [part for _, part in self.history]
            self.history.clear()
        self.parts.append(block)
        self.quiet = 0 if speaking else self.quiet + 1
        if self.quiet >= self.silence_blocks or len(self.parts) >= self.max_blocks:
            return self._emit()
        return []

    def _emit(self):
        if not self.parts:
            return []
        pcm = np.concatenate(self.parts)
        segment = SpeechSegment(self.source, self.start,
                                self.start + len(pcm) / 16000, pcm)
        self.parts = []
        self.quiet = 0
        return [segment]

    def flush(self):
        results = []
        if self.pending.size:
            # Padding is only for inference; remove it from the resulting audio.
            size = self.pending.size
            padded = np.pad(self.pending, (0, 512 - size))
            results.extend(self._block(self.pending_start, padded))
            if self.parts:
                self.parts[-1] = self.parts[-1][:size]
            elif results:
                results[-1].pcm = results[-1].pcm[:-(512 - size)] if size < 512 else results[-1].pcm
                results[-1].end = results[-1].start + len(results[-1].pcm) / 16000
        self.pending = np.empty(0, np.float32)
        results.extend(self._emit())
        self.history.clear()
        self.expected = None
        return results
