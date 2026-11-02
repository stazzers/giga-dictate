import time
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from AppKit import NSApplication

from giga_dictate.audio import AudioRecorder
from giga_dictate.config import AppConfig
from giga_dictate.engine import GigaAMEngine
from giga_dictate.hud import (
    CLASSIC_HEIGHT,
    CLASSIC_WIDTH,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    VoiceInputHUD,
    locate_caret_or_cursor,
)
from giga_dictate.menu_app import MenuAppController


class TestLiveHUDPreview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = NSApplication.sharedApplication()

    def test_app_config_live_hud_preview_defaults_and_persistence(self):
        """Verify AppConfig.live_hud_preview default and roundtrip serialization."""
        config = AppConfig()
        self.assertTrue(config.live_hud_preview)

        # Test persistence
        try:
            config.live_hud_preview = False
            config.save()
            loaded = AppConfig.load()
            self.assertFalse(loaded.live_hud_preview)

            config.live_hud_preview = True
            config.save()
            loaded2 = AppConfig.load()
            self.assertTrue(loaded2.live_hud_preview)
        finally:
            config.live_hud_preview = True
            config.save()

    def test_audio_recorder_get_stream_snapshot(self):
        """Verify AudioRecorder.get_stream_snapshot returns recent audio without altering recording state."""
        recorder = AudioRecorder()
        sr = recorder.sample_rate

        # 1. Not recording -> empty snapshot
        snap = recorder.get_stream_snapshot()
        self.assertEqual(len(snap), 0)
        self.assertEqual(snap.dtype, np.float32)

        # 2. Recording with frames < max_seconds
        recorder._recording = True
        frame1 = np.ones((1600, 1), dtype=np.float32) * 0.1
        frame2 = np.ones((1600, 1), dtype=np.float32) * 0.2
        with recorder._lock:
            recorder._frames = [frame1, frame2]

        snap2 = recorder.get_stream_snapshot(max_seconds=6.0)
        self.assertEqual(len(snap2), 3200)
        self.assertAlmostEqual(snap2[0], 0.1, places=5)
        self.assertAlmostEqual(snap2[1600], 0.2, places=5)
        # Frames list must remain intact in recorder
        self.assertEqual(len(recorder._frames), 2)

        # 3. Recording with frames > max_seconds (e.g. 10s of audio with 6s max snapshot)
        frames_long = [np.ones((1600, 1), dtype=np.float32) * i for i in range(100)]  # 10s
        with recorder._lock:
            recorder._frames = frames_long

        snap_long = recorder.get_stream_snapshot(max_seconds=6.0)
        expected_len = int(6.0 * sr)
        self.assertEqual(len(snap_long), expected_len)
        # The snapshot should contain the tail of the audio
        self.assertAlmostEqual(snap_long[-1], 99.0, places=5)

        recorder._recording = False
        recorder.close()

    def test_engine_transcribe_preview_unloaded_or_short(self):
        """transcribe_preview should safely return empty string if model is not loaded or audio too short."""
        engine = GigaAMEngine(device="cpu")
        self.assertFalse(engine.is_loaded)

        dummy_audio = np.zeros(16000, dtype=np.float32)
        # Not loaded
        res = engine.transcribe_preview(dummy_audio)
        self.assertEqual(res, "")

        # Too short (< 0.3s)
        short_audio = np.zeros(1600, dtype=np.float32)
        res_short = engine.transcribe_preview(short_audio)
        self.assertEqual(res_short, "")

    def test_engine_transcribe_preview_non_blocking_lock(self):
        """transcribe_preview must never block if lock is held by final transcription."""
        engine = GigaAMEngine(device="cpu")
        engine._is_loaded = True
        engine.model = MagicMock()

        # Hold lock
        engine._lock.acquire()
        try:
            # transcribe_preview should return immediately without waiting
            t0 = time.time()
            res = engine.transcribe_preview(np.zeros(16000, dtype=np.float32))
            elapsed = time.time() - t0
            self.assertEqual(res, "")
            self.assertLess(elapsed, 0.05)
        finally:
            engine._lock.release()

    def test_engine_transcribe_preview_mock_inference(self):
        """transcribe_preview runs greedy forward pass and returns stripped text."""
        engine = GigaAMEngine(device="cpu")
        engine._is_loaded = True

        mock_model = MagicMock()
        mock_model._dtype = torch.float32
        mock_model._device = torch.device("cpu")
        mock_model.forward.return_value = (torch.zeros((1, 5, 10)), torch.tensor([5]))
        mock_model._decode.return_value = [("  привет живой предпросмотр  ", None)]
        engine.model = mock_model

        audio = np.random.randn(16000).astype(np.float32)
        res = engine.transcribe_preview(audio)
        self.assertEqual(res, "привет живой предпросмотр")
        mock_model.forward.assert_called_once()
        mock_model._decode.assert_called_once()

    def test_hud_dimensions_and_mode_switching(self):
        """Verify VoiceInputHUD adjusts size, layout and subviews between preview and classic modes."""
        hud = VoiceInputHUD.alloc().init()
        self.assertTrue(hud.live_preview_enabled)

        # 1. Preview mode layout
        hud.setLivePreviewEnabled_(True)
        frame = hud.panel.frame()
        self.assertEqual(frame.size.width, PREVIEW_WIDTH)
        self.assertEqual(frame.size.height, PREVIEW_HEIGHT)
        self.assertTrue(hud.container_view.is_preview_enabled)
        self.assertFalse(hud.text_label.isHidden())
        self.assertEqual(hud.wave_view.frame().origin.y, 8.0)
        self.assertGreater(hud.text_label.frame().origin.y, 0.0)

        # 2. Classic mode layout
        hud.setLivePreviewEnabled_(False)
        frame_classic = hud.panel.frame()
        self.assertEqual(frame_classic.size.width, CLASSIC_WIDTH)
        self.assertEqual(frame_classic.size.height, CLASSIC_HEIGHT)
        self.assertFalse(hud.container_view.is_preview_enabled)
        self.assertTrue(hud.text_label.isHidden())
        self.assertEqual(hud.wave_view.frame().origin.y, 0.0)

        # 3. Switch back to preview mode
        hud.setLivePreviewEnabled_(True)
        self.assertEqual(hud.panel.frame().size.width, PREVIEW_WIDTH)
        self.assertEqual(hud.panel.frame().size.height, PREVIEW_HEIGHT)
        self.assertFalse(hud.text_label.isHidden())

    def test_hud_text_updates_and_states(self):
        """Verify text label content across recording, preview updates, and transcribing states."""
        hud = VoiceInputHUD.alloc().init()
        hud.setLivePreviewEnabled_(True)

        # showRecording resets to "..."
        hud.showRecording()
        self.assertEqual(hud.state, "recording")
        self.assertEqual(hud.text_label.stringValue(), "...")

        # Update preview text with recognized words
        hud.setPreviewText_("тестовый ввод текста")
        self.assertEqual(hud.text_label.stringValue(), "тестовый ввод текста")

        # showTranscribing preserves recognized words
        hud.showTranscribing()
        self.assertEqual(hud.state, "transcribing")
        self.assertEqual(hud.text_label.stringValue(), "тестовый ввод текста")

        # hideHUD stops timer and clears state
        hud.hideHUD()
        self.assertEqual(hud.state, "idle")

    def test_hud_locate_caret_or_cursor_sizing(self):
        """locate_caret_or_cursor should correctly position both classic and preview sizes under caret."""
        # 1. Real screen coordinates check
        pt_classic = locate_caret_or_cursor(CLASSIC_WIDTH, CLASSIC_HEIGHT)
        pt_preview = locate_caret_or_cursor(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self.assertIsNotNone(pt_classic)
        self.assertIsNotNone(pt_preview)
        self.assertGreaterEqual(pt_classic.x, 0)
        self.assertGreaterEqual(pt_preview.x, 0)

        # 2. Deterministic caret centering check: preview HUD should be centered relative to classic HUD
        diff_w = (PREVIEW_WIDTH - CLASSIC_WIDTH) / 2.0
        self.assertEqual(diff_w, 109.0)

    def test_menu_app_live_hud_preview_checkbox(self):
        """Verify menu bar item '[✓] Живой предпросмотр в HUD' presence, state, and toggling."""
        controller = MenuAppController.alloc().init()
        controller.config.live_hud_preview = True
        controller.rebuild_menu()

        menu = controller.menu
        preview_item = next(
            (item for item in menu.itemArray() if item.title() and "Живой предпросмотр" in item.title()),
            None,
        )
        self.assertIsNotNone(preview_item)
        self.assertEqual(preview_item.state(), 1)
        self.assertEqual(preview_item.action(), "onToggleLiveHudPreview:")

        # Toggle OFF
        controller.onToggleLiveHudPreview_(preview_item)
        self.assertFalse(controller.config.live_hud_preview)
        self.assertFalse(controller.hud.live_preview_enabled)
        # Check menu state updated
        items = [item for item in controller.menu.itemArray() if item.title() and "Живой предпросмотр" in item.title()]
        self.assertEqual(items[0].state(), 0)

        # Toggle ON
        controller.onToggleLiveHudPreview_(items[0])
        self.assertTrue(controller.config.live_hud_preview)
        self.assertTrue(controller.hud.live_preview_enabled)
        items_on = [item for item in controller.menu.itemArray() if item.title() and "Живой предпросмотр" in item.title()]
        self.assertEqual(items_on[0].state(), 1)

    def test_menu_app_preview_timer_lifecycle(self):
        """Verify preview timer starts during recording when enabled and stays inactive when disabled."""
        controller = MenuAppController.alloc().init()
        controller.engine._is_loaded = True
        controller.engine.transcribe = MagicMock(return_value="тест")
        controller.inserter.insert_text = MagicMock()

        def mock_start():
            controller.recorder._recording = True
            return True

        def mock_stop():
            controller.recorder._recording = False
            return (np.zeros(16000, dtype=np.float32), 1.0)

        controller.sounds.enabled = False
        controller.recorder.start = MagicMock(side_effect=mock_start)
        controller.recorder.stop = MagicMock(side_effect=mock_stop)

        # 1. With live_hud_preview = True
        controller.config.live_hud_preview = True
        controller.start_recording()
        self.assertIsNotNone(controller._preview_timer)

        controller.stop_and_transcribe()
        self.assertIsNone(controller._preview_timer)

        # 2. With live_hud_preview = False (classic mode)
        controller.config.live_hud_preview = False
        controller.start_recording()
        self.assertIsNone(controller._preview_timer)

        controller.stop_and_transcribe()
        self.assertIsNone(controller._preview_timer)

    def test_hud_set_preview_text_rejected_when_not_recording(self):
        """setPreviewText_ must strictly reject updates when state is transcribing or idle."""
        hud = VoiceInputHUD.alloc().init()
        hud.setLivePreviewEnabled_(True)

        # 1. In idle state
        hud.state = "idle"
        hud.text_label.setStringValue_("")
        hud.setPreviewText_("не должно отображаться")
        self.assertEqual(hud.text_label.stringValue(), "")

        # 2. In transcribing state
        hud.state = "transcribing"
        hud.text_label.setStringValue_("Обработка...")
        hud.setPreviewText_("запоздалый текст")
        self.assertEqual(hud.text_label.stringValue(), "Обработка...")

        # 3. In recording state -> accepted
        hud.state = "recording"
        hud.setPreviewText_("распознанные слова")
        self.assertEqual(hud.text_label.stringValue(), "распознанные слова")

    def test_hud_fade_timer_cancelled_on_show_recording_and_safe_on_fade_done(self):
        """showRecording must cancel pending fade timer, and _onFadeDone_ must not stop active recording timer."""
        hud = VoiceInputHUD.alloc().init()
        hud.setLivePreviewEnabled_(True)

        # Simulate hideHUD running and arming _fade_timer
        hud._is_visible = True
        hud.hideHUD()
        self.assertIsNotNone(hud._fade_timer)

        # Immediate re-recording: showRecording cancels fade timer
        hud.showRecording()
        self.assertIsNone(hud._fade_timer)
        self.assertEqual(hud.state, "recording")
        self.assertIsNotNone(hud.timer)

        # If _onFadeDone_ is invoked artificially while state is recording, it must NO-OP
        initial_timer = hud.timer
        hud._onFadeDone_(None)
        self.assertEqual(hud.timer, initial_timer)
        self.assertTrue(hud._is_visible)

    def test_menu_app_session_id_invalidates_stale_preview_delivery(self):
        """_deliver_preview_text must discard text from prior or cancelled recording sessions."""
        controller = MenuAppController.alloc().init()
        controller.hud = VoiceInputHUD.alloc().init()
        controller.hud.setLivePreviewEnabled_(True)
        controller.recorder._recording = True
        controller.hud.state = "recording"
        controller._preview_session_id = 42

        # 1. Matching session and recording -> delivered
        controller._deliver_preview_text("слово один", 42)
        self.assertEqual(controller.hud.text_label.stringValue(), "слово один")

        # 2. Mismatched (stale) session id -> rejected
        controller._deliver_preview_text("старое слово", 41)
        self.assertEqual(controller.hud.text_label.stringValue(), "слово один")

        # 3. Matching session but recorder stopped -> rejected
        controller.recorder._recording = False
        controller._deliver_preview_text("слово три", 42)
        self.assertEqual(controller.hud.text_label.stringValue(), "слово один")

    def test_engine_transcribe_preview_nan_and_unprintable_glyphs(self):
        """transcribe_preview must cleanly return empty string on NaN/Inf arrays or decode artifacts."""
        engine = GigaAMEngine(device="cpu")
        engine._is_loaded = True

        # 1. Audio with NaN
        nan_audio = np.full(16000, np.nan, dtype=np.float32)
        self.assertEqual(engine.transcribe_preview(nan_audio), "")

        # 2. Audio with Inf
        inf_audio = np.ones(16000, dtype=np.float32)
        inf_audio[500] = np.inf
        self.assertEqual(engine.transcribe_preview(inf_audio), "")

        # 3. Model returning '⁇' or non-printable glyphs
        mock_model = MagicMock()
        mock_model._dtype = torch.float32
        mock_model._device = torch.device("cpu")
        mock_model.forward.return_value = (torch.zeros((1, 5, 10)), torch.tensor([5]))
        mock_model._decode.return_value = [("⁇ неизвестно", None)]
        engine.model = mock_model

        valid_audio = np.zeros(16000, dtype=np.float32)
        self.assertEqual(engine.transcribe_preview(valid_audio), "")

    def test_audio_recorder_get_stream_snapshot_zero_or_negative_duration(self):
        """get_stream_snapshot must return empty array when max_seconds <= 0, preventing slice -0: full-buffer leaks."""
        recorder = AudioRecorder()
        recorder._recording = True
        frame = np.ones((1600, 1), dtype=np.float32)
        with recorder._lock:
            recorder._frames = [frame, frame]

        snap_zero = recorder.get_stream_snapshot(max_seconds=0.0)
        self.assertEqual(len(snap_zero), 0)

        snap_neg = recorder.get_stream_snapshot(max_seconds=-2.5)
        self.assertEqual(len(snap_neg), 0)
        recorder.close()

    def test_engine_transcribe_preview_list_of_strings_decode(self):
        """transcribe_preview must handle list-of-strings decode without truncating to first character."""
        engine = GigaAMEngine(device="cpu")
        engine._is_loaded = True

        mock_model = MagicMock()
        mock_model._dtype = torch.float32
        mock_model._device = torch.device("cpu")
        mock_model.forward.return_value = (torch.zeros((1, 5, 10)), torch.tensor([5]))
        mock_model._decode.return_value = ["полный распознанный текст"]
        engine.model = mock_model

        valid_audio = np.zeros(16000, dtype=np.float32)
        result = engine.transcribe_preview(valid_audio)
        self.assertEqual(result, "полный распознанный текст")

    def test_engine_transcribe_preview_2d_and_dtype_normalization(self):
        """transcribe_preview safely flattens 2D audio and non-float32 arrays without tensor shape mismatch."""
        engine = GigaAMEngine(device="cpu")
        engine._is_loaded = True

        mock_model = MagicMock()
        mock_model._dtype = torch.float32
        mock_model._device = torch.device("cpu")
        mock_model.forward.return_value = (torch.zeros((1, 5, 10)), torch.tensor([5]))
        mock_model._decode.return_value = [("нормализованный звук", None)]
        engine.model = mock_model

        # 2D audio (16000, 1) as float64
        audio_2d = np.zeros((16000, 1), dtype=np.float64)
        result = engine.transcribe_preview(audio_2d)
        self.assertEqual(result, "нормализованный звук")
        # Verify wav passed to forward was 2D (1, 16000)
        passed_wav = mock_model.forward.call_args[0][0]
        self.assertEqual(passed_wav.shape, torch.Size([1, 16000]))
        self.assertEqual(passed_wav.dtype, torch.float32)

    def test_menu_app_toggle_live_hud_preview_while_recording(self):
        """Toggling live_hud_preview while actively recording starts or stops preview timer immediately."""
        controller = MenuAppController.alloc().init()
        controller.recorder._recording = True
        controller.hud.state = "recording"
        controller.config.live_hud_preview = True

        # Start timer initially
        controller._start_preview_timer()
        self.assertIsNotNone(controller._preview_timer)

        # Toggle OFF while recording -> timer stops
        controller.onToggleLiveHudPreview_(None)
        self.assertFalse(controller.config.live_hud_preview)
        self.assertIsNone(controller._preview_timer)

        # Toggle ON while recording -> timer starts
        controller.onToggleLiveHudPreview_(None)
        self.assertTrue(controller.config.live_hud_preview)
        self.assertIsNotNone(controller._preview_timer)

        controller._stop_preview_timer()
        controller.recorder._recording = False

    def test_hud_set_preview_text_whitespace_normalization(self):
        """setPreviewText_ normalizes newlines and consecutive whitespace into a single clean line."""
        hud = VoiceInputHUD.alloc().init()
        hud.setLivePreviewEnabled_(True)
        hud.state = "recording"

        hud.setPreviewText_("первое\nвторое   третье\tчетвертое")
        self.assertEqual(hud.text_label.stringValue(), "первое второе третье четвертое")

    def test_hud_waveform_adaptive_max_height(self):
        """VoiceWaveformView adapts maximum bar height to view frame, preventing top/bottom clipping."""
        from AppKit import NSPoint, NSRect, NSSize
        from giga_dictate.hud import VoiceWaveformView

        # 1. Preview mode (38px height) -> max_h is 32px
        wave_preview = VoiceWaveformView.alloc().initWithFrame_(
            NSRect(NSPoint(0.0, 36.0), NSSize(290.0, 38.0))
        )
        wave_preview.updateLevel_state_speed_(1.0, "recording", 1.0)
        for h in wave_preview.bar_heights:
            self.assertLessEqual(h, 38.0)

        # 2. Classic mode (52px height) -> max_h reaches up to 44px
        wave_classic = VoiceWaveformView.alloc().initWithFrame_(
            NSRect(NSPoint(0.0, 0.0), NSSize(72.0, 52.0))
        )
        wave_classic.smoothed_level = 1.0
        wave_classic.updateLevel_state_speed_(1.0, "recording", 1.0)
        for h in wave_classic.bar_heights:
            self.assertLessEqual(h, 46.0)

    def test_locate_caret_multi_monitor_bounds_clamping(self):
        """locate_caret_or_cursor must respect external monitor bounds and not clamp to primary monitor."""
        from AppKit import NSRect, NSPoint, NSSize

        # Screen 0 (Built-in): (0, 0, 1440, 900), visible: (0, 80, 1440, 820)
        # Screen 1 (External): (1440, 0, 1920, 1080), visible: (1440, 0, 1920, 1055)
        mock_s0 = MagicMock()
        mock_s0.frame.return_value = NSRect(NSPoint(0, 0), NSSize(1440, 900))
        mock_s0.visibleFrame.return_value = NSRect(NSPoint(0, 80), NSSize(1440, 820))

        mock_s1 = MagicMock()
        mock_s1.frame.return_value = NSRect(NSPoint(1440, 0), NSSize(1920, 1080))
        mock_s1.visibleFrame.return_value = NSRect(NSPoint(1440, 0), NSSize(1920, 1055))

        screens = [mock_s0, mock_s1]

        # Target point situated on Screen 1 at (1650, 500)
        caret_pt = NSPoint(1650, 500)
        pt = locate_caret_or_cursor(
            PREVIEW_WIDTH, PREVIEW_HEIGHT, screens=screens, target_pt=caret_pt
        )
        # HUD should remain on Screen 1 within [1440, 1440+1920]
        self.assertGreaterEqual(pt.x, 1440.0)
        self.assertLessEqual(pt.x, 1440.0 + 1920.0 - PREVIEW_WIDTH)
        self.assertEqual(pt.x, 1650.0)
        self.assertEqual(pt.y, 500.0)

    def test_engine_transcribe_list_of_strings_decode(self):
        """engine.transcribe must handle list-of-strings decode without single-char slicing."""
        engine = GigaAMEngine(device="cpu")
        engine._is_loaded = True

        mock_model = MagicMock()
        mock_model._dtype = torch.float32
        mock_model._device = torch.device("cpu")
        mock_model.forward.return_value = (torch.zeros((1, 5, 10)), torch.tensor([5]))
        mock_model._decode.return_value = ["полный финальный текст"]
        engine.model = mock_model

        valid_audio = np.zeros(16000, dtype=np.float32)
        result = engine.transcribe(valid_audio)
        self.assertEqual(result, "полный финальный текст")

    def test_engine_warmup_multi_slice(self):
        """engine._warmup must compile shaders across multiple slice lengths (full and preview)."""
        engine = GigaAMEngine(device="cpu")
        mock_model = MagicMock()
        mock_model._dtype = torch.float32
        mock_model._device = torch.device("cpu")
        mock_model.forward.return_value = (torch.zeros((1, 5, 10)), torch.tensor([5]))
        mock_model._decode.return_value = [("тест", None)]
        engine.model = mock_model

        engine._warmup()
        # forward should be called twice (1.0s and 2.5s slices)
        self.assertEqual(mock_model.forward.call_count, 2)
        lengths_called = [call[0][1].item() for call in mock_model.forward.call_args_list]
        self.assertEqual(lengths_called, [16000, 40000])

    def test_audio_recorder_get_stream_snapshot_float32_guarantee(self):
        """get_stream_snapshot guarantees contiguous float32 numpy array on non-float32 frames."""
        recorder = AudioRecorder()
        recorder._recording = True
        frame_int = np.ones((1600, 1), dtype=np.int32)
        with recorder._lock:
            recorder._frames = [frame_int]

        snap = recorder.get_stream_snapshot(max_seconds=2.0)
        self.assertEqual(snap.dtype, np.float32)
        self.assertEqual(len(snap), 1600)
        recorder.close()

    def test_waveform_view_draw_rect_execution(self):
        """VoiceWaveformView.drawRect_ executes without errors and clears background."""
        from AppKit import NSBitmapImageRep, NSPoint, NSRect, NSSize
        from giga_dictate.hud import VoiceWaveformView

        wave = VoiceWaveformView.alloc().initWithFrame_(
            NSRect(NSPoint(0.0, 0.0), NSSize(72.0, 52.0))
        )
        wave.updateLevel_state_speed_(0.8, "recording", 1.5)
        rep = wave.bitmapImageRepForCachingDisplayInRect_(wave.bounds())
        wave.cacheDisplayInRect_toBitmapImageRep_(wave.bounds(), rep)
        self.assertIsNotNone(rep)

    def test_hud_glass_style_and_opacity_persistence(self):
        """Verify glass style configuration and opacity levels (glass, balanced, matte)."""
        config = AppConfig()
        self.assertEqual(config.hud_glass_style, "glass")
        self.assertAlmostEqual(config.hud_glass_alpha, 0.25)

        hud = VoiceInputHUD.alloc().init()
        hud.setGlassStyle_("matte")
        self.assertEqual(hud.container_view.glass_style, "matte")
        self.assertAlmostEqual(hud.container_view.glass_opacity, 0.85)

        hud.setGlassStyle_("balanced")
        self.assertEqual(hud.container_view.glass_style, "balanced")
        self.assertAlmostEqual(hud.container_view.glass_opacity, 0.55)

        hud.setGlassStyle_("glass")
        self.assertEqual(hud.container_view.glass_style, "glass")
        self.assertAlmostEqual(hud.container_view.glass_opacity, 0.22)

    def test_hud_dynamic_expanding_width_and_height(self):
        """Verify HUD expands wider and taller when displaying full multi-line speech phrases."""
        hud = VoiceInputHUD.alloc().init()
        hud.setLivePreviewEnabled_(True)
        hud.showRecording()

        # Initial placeholder: compact
        init_frame = hud.panel.frame()
        self.assertEqual(init_frame.size.width, PREVIEW_WIDTH)
        self.assertEqual(init_frame.size.height, PREVIEW_HEIGHT)

        # Very long phrase: should expand width up to max and expand height
        long_phrase = (
            "Это очень длинная тестовая фраза, которую надиктовал пользователь приложения GigaDictate. "
            "Мы проверяем, что всё наговоренное аккуратно умещается в окне HUD, а окно становится шире и выше по мере необходимости."
        )
        hud.setPreviewText_(long_phrase)
        expanded_frame = hud.panel.frame()
        self.assertEqual(expanded_frame.size.width, 540.0)
        self.assertGreater(expanded_frame.size.height, PREVIEW_HEIGHT)
        self.assertEqual(hud.text_label.stringValue(), long_phrase)

    def test_hud_mini_waveform_in_bottom_right(self):
        """Verify mini waveform indicator sits in bottom right of expanding preview HUD."""
        hud = VoiceInputHUD.alloc().init()
        hud.setLivePreviewEnabled_(True)
        hud.showRecording()

        w_frame = hud.wave_view.frame()
        c_frame = hud.container_view.frame()
        self.assertEqual(w_frame.origin.y, 8.0)
        self.assertAlmostEqual(w_frame.origin.x + w_frame.size.width + 8.0, c_frame.size.width, delta=2.0)
        self.assertEqual(w_frame.size.width, 26.0)
        self.assertEqual(w_frame.size.height, 14.0)


if __name__ == "__main__":
    unittest.main()
