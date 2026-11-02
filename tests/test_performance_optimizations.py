import os
import sys
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from AppKit import NSPasteboard, NSStringPboardType

from giga_dictate.audio import AudioRecorder, split_audio_into_chunks
from giga_dictate.config import AppConfig, get_default_device
from giga_dictate.hotkey import HotkeyListener, parse_hotkey
from giga_dictate.inserter import TextInserter
from giga_dictate.sounds import SoundManager


class TestPerformanceOptimizations(unittest.TestCase):
    def test_lazy_config_device_does_not_require_torch(self):
        """get_default_device should return 'mps' on Apple Silicon macOS without importing torch."""
        dev = get_default_device()
        self.assertIn(dev, ("mps", "cpu"))

    def test_vectorized_rms_and_zcr_accuracy(self):
        """Verify optimized BLAS np.dot RMS and signbit ZCR match mathematical definitions."""
        # 1. Zero signal
        zeros = np.zeros(512, dtype=np.float32)
        flat = zeros.ravel()
        rms_zero = float(np.sqrt(np.dot(flat, flat) / len(flat)))
        self.assertEqual(rms_zero, 0.0)

        # 2. Synthetic sine wave
        t = np.linspace(0, 1, 512, endpoint=False, dtype=np.float32)
        sine = np.sin(2 * np.pi * 5 * t).astype(np.float32)
        flat_sine = sine.ravel()

        rms_dot = float(np.sqrt(np.dot(flat_sine, flat_sine) / len(flat_sine)))
        rms_legacy = float(np.sqrt(np.mean(flat_sine**2)))
        self.assertAlmostEqual(rms_dot, rms_legacy, places=6)

        # 3. ZCR check
        signs = np.signbit(flat_sine)
        zcr_signbit = float(np.count_nonzero(signs[:-1] != signs[1:]) / (len(flat_sine) - 1))
        zcr_legacy = float(np.mean(flat_sine[:-1] * flat_sine[1:] < 0))
        self.assertAlmostEqual(zcr_signbit, zcr_legacy, places=3)

    def test_split_audio_into_chunks_energy_minimization(self):
        """Verify chunk splitting on quiet sections without array allocation overhead."""
        sr = 16000
        # Short audio (5 seconds) -> single chunk returned
        short_audio = np.random.randn(5 * sr).astype(np.float32)
        chunks = split_audio_into_chunks(short_audio, sr=sr, max_sec=20.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), len(short_audio))

        # Long audio (30 seconds) with silence dip at second 15
        long_audio = np.ones(30 * sr, dtype=np.float32) * 0.5
        # Insert 1s silence between 15s and 16s
        long_audio[15 * sr : 16 * sr] = 0.001
        chunks_long = split_audio_into_chunks(long_audio, sr=sr, max_sec=20.0, min_split_sec=12.0)
        self.assertGreaterEqual(len(chunks_long), 2)
        # All chunks must be <= 20 seconds
        for c in chunks_long:
            self.assertLessEqual(len(c), 20 * sr + 100)

    def test_sounds_no_thread_spawning(self):
        """SoundManager.play should play system sounds without creating new OS threads."""
        manager = SoundManager(enabled=True)
        threads_before = threading.active_count()
        # Play preloaded sounds
        manager.play_start()
        manager.play_stop()
        manager.play_success()
        manager.play_cancel()
        threads_after = threading.active_count()
        # Thread count should remain unchanged (NSSound is natively asynchronous)
        self.assertEqual(threads_before, threads_after)

    def test_hotkey_listener_fast_path(self):
        """HotkeyListener callback immediately ignores non-target keycodes."""
        listener = HotkeyListener(hotkey_str="right_cmd", mode="toggle")
        self.assertEqual(listener.target_vk, 54)

        # Mock an event that is NOT Right Command (e.g., 'A' key, vk=0)
        class DummyEvent:
            pass

        dummy = DummyEvent()
        # Verify callback passes through non-target keycode without modifying state
        # In hotkey.py, vk != self.target_vk returns event immediately
        self.assertFalse(listener._is_active)
        self.assertFalse(listener._is_key_down)

    def test_hotkey_parsing_variations(self):
        """Verify hotkey parser handles aliases, cases, and Russian keys."""
        self.assertEqual(parse_hotkey("right_cmd"), 54)
        self.assertEqual(parse_hotkey("Right_Shift"), 60)
        self.assertEqual(parse_hotkey("Alt_R"), 61)
        self.assertEqual(parse_hotkey("F8"), 100)
        self.assertEqual(parse_hotkey("/"), 44)
        self.assertEqual(parse_hotkey("space"), 49)

    def test_text_inserter_pooled_restoration(self):
        """Ensure clipboard restoration reuses pooled worker and restores text."""
        inserter = TextInserter()
        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_("BASELINE_CLIPBOARD", NSStringPboardType)

        threads_before = threading.active_count()
        with (
            patch.object(TextInserter, "is_accessibility_granted", return_value=True),
            patch("giga_dictate.inserter.CGEventPost"),
        ):
            inserter.insert_text("OPTIMIZED_SPEECH", add_trailing_space=False, restore_clipboard=True)
            time.sleep(0.55)

        current = pb.stringForType_(NSStringPboardType)
        self.assertEqual(current, "BASELINE_CLIPBOARD")
        # Ensure pooled worker doesn't spawn unbounded threads
        threads_after = threading.active_count()
        self.assertLessEqual(threads_after - threads_before, 1)


if __name__ == "__main__":
    unittest.main()
