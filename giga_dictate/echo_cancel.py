import numpy as np

SAMPLE_RATE = 16000
MIN_DELAY = 128
MAX_CORR_LAG = 6400
FAR_RETAIN = 2 * SAMPLE_RATE + MAX_CORR_LAG
FILTER_LEN = 3200
HOLD_SAMPLES = 8000
BLOCK = 160
CORR_THRESHOLD = 0.35
SILENCE_STD = 1e-4
NLMS_MU = 0.25
MAX_FILTER = 1.2
MAX_RESIDUAL = 1.0
EPS = 1e-8


def _best_lag(near, start_sample, read_far, lag_min=MIN_DELAY, lag_max=MAX_CORR_LAG):
    n = len(near)
    if n < 64:
        return 0.0, MIN_DELAY
    if float(np.std(near)) < SILENCE_STD:
        return 0.0, MIN_DELAY
    best, best_lag = 0.0, MIN_DELAY
    step = 16 if lag_max > 512 else 1
    for lag in range(lag_min, lag_max + 1, step):
        far = read_far(start_sample - lag, n)
        if float(np.std(far)) < SILENCE_STD:
            continue
        cn = near - np.mean(near)
        cf = far - np.mean(far)
        denom = float(np.linalg.norm(cn) * np.linalg.norm(cf)) + EPS
        c = abs(float(np.dot(cn, cf) / denom))
        if c > best:
            best, best_lag = c, lag
    lo = max(lag_min, best_lag - step)
    hi = min(lag_max, best_lag + step)
    for lag in range(lo, hi + 1):
        far = read_far(start_sample - lag, n)
        if float(np.std(far)) < SILENCE_STD:
            continue
        cn = near - np.mean(near)
        cf = far - np.mean(far)
        denom = float(np.linalg.norm(cn) * np.linalg.norm(cf)) + EPS
        c = abs(float(np.dot(cn, cf) / denom))
        if c > best:
            best, best_lag = c, lag
    return best, best_lag


class EchoCanceller:
    def __init__(self):
        self._w = np.zeros(FILTER_LEN, dtype=np.float32)
        self._adapted = False
        self._echo_delay = MIN_DELAY
        self._coarse_gain = 0.0
        self._far_base = None
        self._far = np.zeros(0, dtype=np.float32)
        self._hold = []
        self._hold_samples = 0
        self._near_total = 0
        self._near_out = 0

    def note_gap(self, source):
        if source == "microphone":
            self._w.fill(0)
            self._adapted = False
            self._coarse_gain = 0.0
            self._echo_delay = MIN_DELAY
            self._hold.clear()
            self._hold_samples = 0
        elif source == "system":
            self._far_base = None
            self._far = np.zeros(0, dtype=np.float32)
            self._w.fill(0)
            self._adapted = False
            self._coarse_gain = 0.0
            self._echo_delay = MIN_DELAY

    def _sec_to_sample(self, start):
        return int(round(start * SAMPLE_RATE))

    def _ensure_far_capacity(self, end_sample):
        if self._far_base is None:
            self._far_base = 0
            self._far = np.zeros(max(1, end_sample), dtype=np.float32)
            return
        cur_end = self._far_base + len(self._far)
        if end_sample > cur_end:
            self._far = np.pad(self._far, (0, end_sample - cur_end))
        elif end_sample < self._far_base:
            shift = self._far_base - end_sample
            self._far = np.pad(self._far, (shift, 0))
            self._far_base = end_sample

    def _trim_far(self):
        if self._far_base is None or len(self._far) <= FAR_RETAIN:
            return
        drop = len(self._far) - FAR_RETAIN
        self._far = self._far[drop:]
        self._far_base += drop

    def _write_far(self, start_sample, pcm):
        pcm = np.asarray(pcm, dtype=np.float32).ravel()
        if not len(pcm):
            return
        end = start_sample + len(pcm)
        if self._far_base is None:
            self._far_base = start_sample
            self._far = pcm.copy()
        else:
            self._ensure_far_capacity(end)
            off = start_sample - self._far_base
            self._far[off:off + len(pcm)] = pcm
        self._trim_far()

    def _far_available_through(self, end_sample):
        if self._far_base is None:
            return False
        return end_sample <= self._far_base + len(self._far)

    def _read_far(self, start_sample, length):
        if length <= 0 or self._far_base is None:
            return np.zeros(length, dtype=np.float32)
        off = start_sample - self._far_base
        if off < 0:
            return np.zeros(length, dtype=np.float32)
        if off >= len(self._far):
            return np.zeros(length, dtype=np.float32)
        chunk = self._far[off:off + length]
        if len(chunk) < length:
            out = np.zeros(length, dtype=np.float32)
            out[:len(chunk)] = chunk
            return out
        return chunk.astype(np.float32, copy=False)

    def _emit(self, start_sec, pcm):
        pcm = np.asarray(pcm, dtype=np.float32).ravel()
        if not np.isfinite(pcm).all():
            pcm = np.nan_to_num(pcm, copy=False)
        self._near_out += len(pcm)
        return (start_sec, pcm)

    def _passthrough(self, start_sec, pcm):
        return [self._emit(start_sec, pcm)]

    def _build_block_matrix(self, start_sample, offset, blen, delay):
        idx = start_sample + offset + np.arange(blen)[:, None] - delay - np.arange(FILTER_LEN)
        base = self._far_base
        if base is None:
            return np.zeros((blen, FILTER_LEN), dtype=np.float32)
        flat = idx.ravel()
        rel = flat - base
        out = np.zeros(flat.shape[0], dtype=np.float32)
        valid = (rel >= 0) & (rel < len(self._far))
        out[valid] = self._far[rel[valid]]
        return out.reshape(blen, FILTER_LEN)

    def _nlms_tail(self, start_sample, residual, delay, allow_adapt):
        n = len(residual)
        w = self._w.astype(np.float64, copy=True)
        adapted = self._adapted
        out = residual.astype(np.float64, copy=True)
        for b0 in range(0, n, BLOCK):
            blen = min(BLOCK, n - b0)
            X = self._build_block_matrix(start_sample, b0, blen, delay)
            d = out[b0:b0 + blen]
            y = X @ w
            e = d - y
            near_pwr = float(np.mean(d * d)) + EPS
            echo_pwr = float(np.mean(y * y)) + EPS
            double_talk = adapted and echo_pwr < 0.2 * near_pwr
            out[b0:b0 + blen] = e if (adapted or allow_adapt) and not (double_talk and not adapted) else d
            if allow_adapt and not double_talk:
                for i in range(blen):
                    x = X[i]
                    xi = float(x @ x) + EPS
                    err = float(np.clip(e[i], -MAX_RESIDUAL, MAX_RESIDUAL))
                    w += (NLMS_MU * err / xi) * x
                w = np.clip(w, -MAX_FILTER, MAX_FILTER)
                adapted = True
        self._w = w.astype(np.float32, copy=False)
        self._adapted = adapted
        return np.clip(out, -MAX_RESIDUAL, MAX_RESIDUAL).astype(np.float32, copy=False)

    def _apply_aec(self, start_sample, near, allow_adapt):
        near = np.asarray(near, dtype=np.float32).ravel()
        n = len(near)
        peak, lag = _best_lag(near, start_sample, self._read_far)
        if peak < CORR_THRESHOLD:
            return near.copy()
        delay = max(MIN_DELAY, lag)
        if not self._adapted or abs(lag - self._echo_delay) < 96:
            self._echo_delay = delay

        aligned = self._read_far(start_sample - self._echo_delay, n)
        denom = float(np.dot(aligned, aligned)) + EPS
        alpha = float(np.dot(near, aligned) / denom)
        alpha = float(np.clip(alpha, -1.2, 1.2))

        double_talk = self._adapted and abs(alpha) < 0.35 * max(abs(self._coarse_gain), 0.05)
        if allow_adapt and not double_talk:
            self._coarse_gain = 0.85 * self._coarse_gain + 0.15 * alpha if self._adapted else alpha
        gain = self._coarse_gain if self._adapted or not allow_adapt else alpha

        if not self._adapted and not allow_adapt:
            return near.copy()

        coarse = near - gain * aligned
        if float(np.std(coarse)) < SILENCE_STD:
            self._adapted = True
            return coarse

        cleaned = self._nlms_tail(start_sample, coarse, self._echo_delay, allow_adapt and not double_talk)
        self._adapted = True
        if float(np.mean(cleaned * cleaned)) > 1.25 * float(np.mean(near * near)) + EPS:
            return near.copy()
        return cleaned

    def _clean_or_raw(self, start_sec, near, allow_adapt):
        if float(np.std(near)) < SILENCE_STD:
            return self._passthrough(start_sec, near)
        start_sample = self._sec_to_sample(start_sec)
        if not self._far_available_through(start_sample + len(near)):
            return self._passthrough(start_sec, near)
        cleaned = self._apply_aec(start_sample, near, allow_adapt)
        return [self._emit(start_sec, cleaned)]

    def _release_hold_overflow(self):
        out = []
        while self._hold_samples > HOLD_SAMPLES and self._hold:
            start_sec, pcm = self._hold.pop(0)
            self._hold_samples -= len(pcm)
            out.extend(self._passthrough(start_sec, pcm))
        return out

    def _try_drain_hold(self):
        out = []
        keep = []
        for start_sec, pcm in self._hold:
            ss = self._sec_to_sample(start_sec)
            if self._far_available_through(ss + len(pcm)):
                out.extend(self._clean_or_raw(start_sec, pcm, allow_adapt=True))
            else:
                keep.append((start_sec, pcm))
        self._hold = keep
        self._hold_samples = sum(len(x[1]) for x in keep)
        return out

    def push_far(self, start, pcm):
        pcm = np.asarray(pcm, dtype=np.float32).ravel()
        if not len(pcm):
            return self._release_hold_overflow()
        self._write_far(self._sec_to_sample(start), pcm)
        out = self._release_hold_overflow()
        out.extend(self._try_drain_hold())
        return out

    def push_near(self, start, pcm):
        pcm = np.asarray(pcm, dtype=np.float32).ravel()
        self._near_total += len(pcm)
        if not len(pcm):
            return self._release_hold_overflow()

        start_sec = float(start)
        end_sample = self._sec_to_sample(start_sec) + len(pcm)

        if float(np.std(pcm)) < SILENCE_STD:
            out = self._passthrough(start_sec, pcm)
            out.extend(self._release_hold_overflow())
            return out

        out = []
        if not self._far_available_through(end_sample):
            if self._hold_samples + len(pcm) <= HOLD_SAMPLES:
                self._hold.append((start_sec, pcm.copy()))
                self._hold_samples += len(pcm)
                out.extend(self._release_hold_overflow())
                return out
            emit_len = len(pcm) + self._hold_samples - HOLD_SAMPLES
            if emit_len > 0:
                raw = pcm[-emit_len:]
                hold_part = pcm[:-emit_len]
                if len(hold_part):
                    self._hold.append((start_sec, hold_part.copy()))
                    self._hold_samples += len(hold_part)
                if len(raw):
                    out.extend(self._passthrough(start_sec + (len(pcm) - len(raw)) / SAMPLE_RATE, raw))
            else:
                self._hold.append((start_sec, pcm.copy()))
                self._hold_samples += len(pcm)
            out.extend(self._release_hold_overflow())
            return out

        out.extend(self._clean_or_raw(start_sec, pcm, allow_adapt=True))
        out.extend(self._release_hold_overflow())
        return out

    def flush(self):
        out = []
        out.extend(self._release_hold_overflow())
        for start_sec, pcm in self._hold:
            ss = self._sec_to_sample(start_sec)
            if float(np.std(pcm)) < SILENCE_STD:
                out.append(self._emit(start_sec, pcm))
            elif self._far_available_through(ss + len(pcm)):
                cleaned = self._apply_aec(ss, pcm, allow_adapt=False)
                out.append(self._emit(start_sec, cleaned))
            else:
                out.append(self._emit(start_sec, pcm))
        self._hold.clear()
        self._hold_samples = 0
        return out