"""Local CAM++ embeddings, conservative online clustering and owner enrollment."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .meeting_types import Speaker, SpeechSegment

MODEL_ID = "iic/speech_campplus_sv_zh_en_16k-common_advanced@v1.0.0"
PREPROCESSING = "kaldi-fbank80-16k-mean-v1"


def unit(vector):
    vector = np.asarray(vector, np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if vector.shape != (192,) or not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError("Некорректный голосовой вектор")
    return vector / norm


class CAMPlusEmbedder:
    def __init__(self, path):
        import onnxruntime as ort
        import kaldi_native_fbank as knf
        self.knf = knf
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), opts,
                                            providers=["CPUExecutionProvider"])

    def __call__(self, pcm):
        options = self.knf.FbankOptions()
        options.frame_opts.dither = 0
        options.frame_opts.samp_freq = 16000
        options.frame_opts.snip_edges = True
        options.mel_opts.num_bins = 80
        options.mel_opts.debug_mel = False
        fbank = self.knf.OnlineFbank(options)
        fbank.accept_waveform(16000, np.asarray(pcm, np.float32).tolist())
        fbank.input_finished()
        if fbank.num_frames_ready < 10:
            raise ValueError("Недостаточно речи для голосового вектора")
        features = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        features -= features.mean(axis=0, keepdims=True)
        vector = self.session.run(None, {self.session.get_inputs()[0].name:
                                        features[None].astype(np.float32)})[0]
        return unit(vector)


class OwnerProfile:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return None
        try:
            if self.path.stat().st_size > 65536:
                raise ValueError()
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data["model"] != MODEL_ID or data["preprocessing"] != PREPROCESSING:
                raise ValueError()
            vectors = np.stack([unit(v) for v in data["vectors"]])
            if not 3 <= len(vectors) <= 16:
                raise ValueError()
            return vectors
        except (ValueError, KeyError, TypeError):
            raise ValueError("Голосовой профиль несовместим или повреждён. Запишите его заново.") from None

    def save(self, vectors):
        vectors = np.stack([unit(v) for v in vectors])
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".voice-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"model": MODEL_ID, "preprocessing": PREPROCESSING,
                           "vectors": vectors.tolist()}, handle)
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def delete(self):
        self.path.unlink(missing_ok=True)


class Enrollment:
    def __init__(self, embedder):
        self.embedder = embedder
        self.vectors = []
        self.speech_seconds = 0.0
        self.samples = 0
        self.clipped = 0

    def feed(self, segment):
        self.speech_seconds += len(segment.pcm) / 16000
        self.samples += len(segment.pcm)
        self.clipped += int(np.count_nonzero(np.abs(segment.pcm) >= .99))
        for offset in range(0, len(segment.pcm) - 32000 + 1, 32000):
            if len(self.vectors) < 16:
                self.vectors.append(self.embedder(segment.pcm[offset:offset + 32000]))

    def finish(self):
        if self.speech_seconds < 8 or len(self.vectors) < 3:
            raise ValueError("Недостаточно чистой речи. Повторите запись, говорите 15–20 секунд.")
        if self.clipped / max(1, self.samples) > .01:
            raise ValueError("Микрофон перегружен. Отодвиньтесь и повторите запись.")
        vectors = np.stack(self.vectors)
        center = unit(vectors.mean(axis=0))
        if float(np.min(vectors @ center)) < .65:
            raise ValueError("Голос нестабилен или слышны другие люди. Повторите запись в тишине.")
        return vectors


class OnlineDiarizer:
    # Initial conservative defaults, exposed for corpus-based calibration.
    def __init__(self, embedder, owner=None, match=.5, new=.3, owner_match=.75):
        self.embedder, self.owner = embedder, owner
        self.owner_center = unit(owner.mean(axis=0)) if owner is not None else None
        self.match, self.new, self.owner_match = match, new, owner_match
        self.centers = {}
        self.counts = {}
        self.speakers = {"unknown": Speaker("unknown", "Спикер не определён")}
        if owner is not None:
            self.speakers["owner"] = Speaker("owner", "Вы")

    def identify(self, vector):
        if self.owner_center is not None and float(self.owner_center @ vector) >= self.owner_match:
            return "owner"
        if self.centers:
            scores = {key: float(center @ vector) for key, center in self.centers.items()}
            best = max(scores, key=scores.get)
            if scores[best] >= self.match:
                count = self.counts[best]
                self.centers[best] = unit(self.centers[best] * min(count, 20) + vector)
                self.counts[best] += 1
                return best
            if scores[best] > self.new:
                return "unknown"
        if len(self.centers) >= 15:
            return "unknown"
        number = len(self.centers) + 1
        key = f"speaker-{number}"
        self.centers[key] = vector.copy()
        self.counts[key] = 1
        self.speakers[key] = Speaker(key, f"Собеседник {number}")
        return key

    def split(self, segment):
        pcm = segment.pcm
        window, hop = 24000, 8000
        if len(pcm) < 16000:
            return [(segment, "unknown")]
        offsets = list(range(0, max(1, len(pcm) - window + 1), hop))
        last = max(0, len(pcm) - window)
        if offsets[-1] != last:
            offsets.append(last)
        labels = [self.identify(self.embedder(pcm[i:i + window])) for i in offsets]
        original = labels.copy()
        for i in range(1, len(labels) - 1):
            if original[i - 1] == original[i + 1]:
                labels[i] = original[i - 1]
        runs = []
        left, current = 0, labels[0]
        for i in range(1, len(labels)):
            if labels[i] != current:
                boundary = min(len(pcm), (offsets[i - 1] + offsets[i] + window) // 2)
                runs.append((left, boundary, current))
                left, current = boundary, labels[i]
        runs.append((left, len(pcm), current))
        return [(SpeechSegment(segment.source, segment.start + a / 16000,
                               segment.start + b / 16000, pcm[a:b]), speaker)
                for a, b, speaker in runs if b > a]
