import unittest

import numpy as np

from giga_dictate.echo_cancel import EchoCanceller, SAMPLE_RATE


def speech_like_noise(rng, length):
    x = rng.standard_normal(length).astype(np.float32)
    window = np.hanning(64).astype(np.float32)
    out = np.convolve(x, window, mode="same")
    std = float(np.std(out)) or 1.0
    return (out / std) * 0.1


def delayed(near_base, delay_samples, gain=1.0):
    out = np.zeros_like(near_base)
    if delay_samples < len(out):
        out[delay_samples:] = gain * near_base[: len(out) - delay_samples]
    return out


def push_chunks(canceller, far, near, chunk=320, lead_far=0):
    emitted = []
    t = 0.0
    step = chunk / SAMPLE_RATE
    n = len(near)
    far_pad = np.zeros(lead_far, dtype=np.float32)
    far_all = np.concatenate([far_pad, far]) if lead_far else far
    for i in range(0, n, chunk):
        size = min(chunk, n - i)
        if i < len(far_all):
            fsize = min(size, len(far_all) - i)
            far_piece = far_all[i:i + fsize]
            if len(far_piece) < size:
                far_piece = np.pad(far_piece, (0, size - len(far_piece)))
        else:
            far_piece = np.zeros(size, dtype=np.float32)
        emitted.extend(canceller.push_far(t, far_piece))
        emitted.extend(canceller.push_near(t, near[i:i + size]))
        t += step
    emitted.extend(canceller.flush())
    return emitted


class EchoCancelTests(unittest.TestCase):
    def test_delayed_echo_at_least_10db(self):
        rng = np.random.default_rng(7)
        far = speech_like_noise(rng, 8 * SAMPLE_RATE)
        echo_delay = 640
        near = delayed(far, echo_delay, 0.6)
        in_pwr = float(np.mean(near[echo_delay + 4000:] ** 2))
        emitted = push_chunks(EchoCanceller(), far, near)
        total = sum(len(p) for _, p in emitted)
        self.assertEqual(total, len(near))
        out = np.concatenate([p for _, p in emitted])
        res_pwr = float(np.mean(out[echo_delay + 8000:] ** 2))
        self.assertLess(res_pwr / in_pwr, 0.1)
        self.assertAlmostEqual(emitted[0][0], 0.0, places=3)

    def test_headphones_preserve_near(self):
        rng = np.random.default_rng(11)
        far = speech_like_noise(rng, 4 * SAMPLE_RATE)
        near = speech_like_noise(rng, 4 * SAMPLE_RATE)
        emitted = push_chunks(EchoCanceller(), far, near)
        out = np.concatenate([p for _, p in emitted])
        cn = near[: len(out)]
        co = out[: len(cn)]
        cn = cn - np.mean(cn)
        co = co - np.mean(co)
        corr = float(np.dot(cn, co) / (np.linalg.norm(cn) * np.linalg.norm(co) + 1e-8))
        self.assertGreater(corr, 0.95)
        self.assertLess(abs(np.mean(co ** 2) - np.mean(cn ** 2)), 0.15 * np.mean(cn ** 2) + 1e-6)

    def test_double_talk_retains_local_voice(self):
        rng = np.random.default_rng(21)
        far = speech_like_noise(rng, 6 * SAMPLE_RATE)
        local = speech_like_noise(rng, 6 * SAMPLE_RATE)
        echo_delay = 640
        near = 0.5 * delayed(far, echo_delay, 1.0) + local
        emitted = push_chunks(EchoCanceller(), far, near)
        out = np.concatenate([p for _, p in emitted])
        tail = slice(echo_delay + 6000, min(len(out), len(local)))
        loc = local[tail]
        och = out[tail]
        loc = loc[: len(och)]
        ln = loc - np.mean(loc)
        on = och - np.mean(och)
        corr_local = float(np.dot(ln, on) / (np.linalg.norm(ln) * np.linalg.norm(on) + 1e-8))
        far_s = far[tail]
        far_s = far_s[: len(loc)]
        fn = far_s - np.mean(far_s)
        corr_far = float(np.dot(fn, ln) / (np.linalg.norm(fn) * np.linalg.norm(ln) + 1e-8))
        self.assertGreater(corr_local, abs(corr_far) + 0.1)
        self.assertGreater(float(np.mean(och ** 2)), 0.4 * float(np.mean(loc ** 2)))

    def test_constant_buffers_not_silenced(self):
        far = np.full(32000, 0.1, np.float32)
        near = np.full(32000, 0.1, np.float32)
        emitted = push_chunks(EchoCanceller(), far, near, chunk=320)
        out = np.concatenate([p for _, p in emitted])
        self.assertGreater(float(np.max(np.abs(out))), 0.05)

    def test_sample_accounting(self):
        rng = np.random.default_rng(3)
        far = speech_like_noise(rng, SAMPLE_RATE * 3)
        near = delayed(far, 480, 0.7)
        c = EchoCanceller()
        emitted = push_chunks(c, far, near, chunk=160)
        total = sum(len(p) for _, p in emitted)
        self.assertEqual(total, len(near))
        self.assertTrue(all(np.isfinite(p).all() for _, p in emitted))

    def test_hold_bound_emits_without_far(self):
        c = EchoCanceller()
        chunk = 320
        seconds = 2.0
        n = int(seconds * SAMPLE_RATE)
        emitted = []
        for i in range(0, n, chunk):
            size = min(chunk, n - i)
            pcm = np.full(size, 0.1, np.float32)
            got = c.push_near(i / SAMPLE_RATE, pcm)
            emitted.extend(got)
        self.assertTrue(emitted)
        tail = c.flush()
        self.assertEqual(sum(len(p) for _, p in emitted) + sum(len(p) for _, p in tail), n)
        all_out = np.concatenate([p for _, p in emitted + tail])
        self.assertGreater(float(np.max(np.abs(all_out))), 0.05)


if __name__ == "__main__":
    unittest.main()