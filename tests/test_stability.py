"""Regression tests for clipboard ownership, session races and bounded preview."""
import logging
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from AppKit import NSPasteboardTypeString

from giga_dictate.audio import AudioRecorder
from giga_dictate.engine import GigaAMEngine
from giga_dictate.events import SYNTHETIC_EVENT_TAG
from giga_dictate.hotkey import HotkeyListener
from giga_dictate.inserter import InsertionResult, TextInserter
from giga_dictate.menu_app import MenuAppController
from giga_dictate.session import DictationSession, DictationState, RecoveryText
from tests.support import FakePasteboardItem, IsolatedTestCase


class TestInsertionTransactions(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.inserter = TextInserter(clock=lambda: self.scheduler.now)
        self.pasteboard.setString_forType_("original", NSPasteboardTypeString)

    def paste(self, text="dictated", **kwargs):
        return self.inserter.insert_text(text, add_trailing_space=False,
                                         restore_clipboard=True, **kwargs)

    def test_complete_after_key_up_and_restore_without_sleep(self):
        result = self.paste()
        self.assertFalse(result.done())
        self.scheduler.step()  # prepare clipboard
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "dictated")
        self.scheduler.step()  # key down
        self.scheduler.step()  # key up
        self.assertFalse(result.done())
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "dictated")
        self.scheduler.run_all()
        self.assertTrue(result.result().event_posted)
        self.assertGreaterEqual(self.scheduler.now, 0.75)
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "original")

    def test_new_user_copy_is_not_overwritten(self):
        result = self.paste()
        for _ in range(3):
            self.scheduler.step()
        self.pasteboard.clearContents()
        self.pasteboard.setString_forType_("new user copy", NSPasteboardTypeString)
        self.scheduler.run_all()
        self.assertTrue(result.result().event_posted)
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "new user copy")

    def test_clipboard_changed_before_paste_aborts_without_overwriting(self):
        result = self.paste()
        self.scheduler.step()
        self.pasteboard.clearContents()
        self.pasteboard.setString_forType_("new user copy", NSPasteboardTypeString)
        self.scheduler.run_all()
        self.assertEqual(result.result().reason, "clipboard_changed")
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "new user copy")

    def test_multiple_items_and_binary_formats_are_restored(self):
        original = [FakePasteboardItem({"public.png": b"\x89PNG\x00", "public.utf8-plain-text": b"caption"}),
                    FakePasteboardItem({"public.rtf": b"{\\rtf1 hello}"})]
        self.pasteboard.writeObjects_(original)
        result = self.paste()
        self.scheduler.run_all()
        self.assertTrue(result.result().event_posted)
        self.assertEqual([i.values for i in self.pasteboard.items], [i.values for i in original])

    def test_copy_during_snapshot_is_not_replaced(self):
        snapshot = self.inserter._snapshot
        def changing_snapshot():
            saved = snapshot()
            self.pasteboard.clearContents()
            self.pasteboard.setString_forType_("new copy", NSPasteboardTypeString)
            return saved
        with patch.object(self.inserter, "_snapshot", side_effect=changing_snapshot):
            result = self.paste()
            self.scheduler.run_all()
        self.assertEqual(result.result().reason, "clipboard_changed")
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "new copy")

    def test_unreadable_snapshot_leaves_clipboard_intact(self):
        with patch.object(self.inserter, "_snapshot", side_effect=RuntimeError("unavailable")):
            result = self.paste()
            self.scheduler.run_all()
        self.assertEqual(result.result().reason, "clipboard_snapshot_failed")
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "original")

    def test_accessibility_denied_preserves_original_clipboard(self):
        with patch.object(TextInserter, "is_accessibility_granted", return_value=False):
            result = self.paste()
        self.assertEqual(result.result().reason, "accessibility_denied")
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "original")

    def test_modifier_timeout_is_nonblocking_and_preserves_clipboard(self):
        with patch("giga_dictate.inserter.CGEventSourceFlagsState", return_value=1 << 20):
            result = self.paste()
            self.assertFalse(result.done())
            self.scheduler.run_all()
        self.assertEqual(result.result().reason, "modifiers_held")
        self.assertGreaterEqual(self.scheduler.now, 1.0)
        self.assertLess(self.scheduler.now, 1.03)
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "original")

    def test_modifiers_can_be_released_before_deadline(self):
        with patch("giga_dictate.inserter.CGEventSourceFlagsState", side_effect=[1 << 20, 0, 0]):
            result = self.paste()
            self.scheduler.run_all()
        self.assertTrue(result.result().event_posted)

    def test_modifier_pressed_after_prepare_aborts(self):
        with patch("giga_dictate.inserter.CGEventSourceFlagsState", side_effect=[0, 1 << 20]):
            result = self.paste()
            self.scheduler.run_all()
        self.assertEqual(result.result().reason, "modifiers_held")
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "original")

    def test_target_change_during_transaction_aborts_and_restores(self):
        with patch.object(self.inserter, "_front_pid", side_effect=[1, 1, 2]):
            result = self.paste()
            self.scheduler.run_all()
        self.assertEqual(result.result().reason, "target_changed")
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "original")

    def test_failed_event_creation_never_replaces_clipboard(self):
        with patch("giga_dictate.inserter.CGEventCreateKeyboardEvent", return_value=None):
            result = self.paste()
            self.scheduler.run_all()
        self.assertEqual(result.result().reason, "event_creation_failed")
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "original")

    def test_clipboard_write_failure_is_reported_and_original_is_restored(self):
        self.pasteboard.fail_write = True
        result = self.paste()
        self.scheduler.run_all()
        self.assertEqual(result.result().reason, "clipboard_write_failed")
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "original")

    def test_event_post_failure_is_not_reported_as_success(self):
        with patch("giga_dictate.inserter.CGEventPost", side_effect=RuntimeError("failure")):
            result = self.paste()
            self.scheduler.run_all()
        self.assertEqual(result.result().status, "error")
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "original")

    def test_overlapping_transaction_is_rejected(self):
        first = self.paste()
        second = self.paste("second")
        self.assertEqual(second.result().reason, "busy")
        self.scheduler.run_all()
        self.assertTrue(first.result().event_posted)

    def test_cancelled_callbacks_cannot_modify_a_new_operation(self):
        old = self.paste("old")
        self.inserter.cancel_pending()
        new = self.paste("new")
        self.scheduler.run_all()
        self.assertEqual(old.result().reason, "cancelled")
        self.assertTrue(new.result().event_posted)
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "original")

    def test_generated_events_are_tagged(self):
        with patch("giga_dictate.inserter.CGEventSetIntegerValueField") as tagged:
            result = self.paste()
            self.scheduler.run_all()
        self.assertTrue(result.result().event_posted)
        self.assertEqual(tagged.call_count, 2)
        self.assertTrue(all(call.args[2] == SYNTHETIC_EVENT_TAG for call in tagged.call_args_list))

    def test_explicit_copy_is_retained_for_manual_paste(self):
        result = self.inserter.copy_text("manual")
        self.scheduler.run_all()
        self.assertEqual(result.status, "copied")
        self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "manual")

    def test_100_sequential_transactions_do_not_duplicate_or_lose_clipboard(self):
        with patch("giga_dictate.inserter.CGEventPost") as post:
            for index in range(100):
                result = self.paste(f"phrase {index}")
                self.scheduler.run_all()
                self.assertTrue(result.result().event_posted)
                self.assertEqual(self.pasteboard.stringForType_(NSPasteboardTypeString), "original")
            self.assertEqual(post.call_count, 200)

    def test_logs_do_not_contain_text_or_clipboard(self):
        with self.assertLogs("giga_dictate.inserter", logging.INFO) as logs:
            result = self.paste("CONFIDENTIAL_PHRASE")
            self.scheduler.run_all()
        self.assertTrue(result.result().event_posted)
        self.assertNotIn("CONFIDENTIAL_PHRASE", "\n".join(logs.output))
        self.assertNotIn("original", "\n".join(logs.output))


class TestSessionAndRecovery(IsolatedTestCase):
    def test_busy_session_rejects_new_recording_without_invalidating(self):
        session = DictationSession()
        self.assertTrue(session.begin())
        generation = session.generation
        for state in (DictationState.RECORDING, DictationState.TRANSCRIBING, DictationState.INSERTING):
            session.state = state
            self.assertFalse(session.begin())
            self.assertEqual(session.generation, generation)
        self.assertTrue(session.finish(generation))
        self.assertTrue(session.begin())
        self.assertFalse(session.finish(generation))
        self.assertEqual(session.state, DictationState.RECORDING)

    def test_shutdown_invalidates_pending_completion(self):
        session = DictationSession()
        session.begin()
        generation = session.generation
        session.invalidate()
        self.assertFalse(session.finish(generation))

    def test_recovery_expires_and_empty_result_does_not_replace_it(self):
        with patch("giga_dictate.session.time.monotonic", return_value=10):
            recovery = RecoveryText()
            recovery.remember("last text")
            recovery.remember("")
            self.assertEqual(recovery.get(), "last text")
            self.assertNotIn("last text", repr(recovery))
        with patch("giga_dictate.session.time.monotonic", return_value=310):
            self.assertEqual(recovery.get(), "")
            self.assertEqual(recovery._text, "")

    def test_synthetic_hotkey_events_are_ignored(self):
        listener = HotkeyListener(hotkey_str="v", mode="toggle", on_start=MagicMock())
        event = object()
        with patch("giga_dictate.hotkey.CGEventGetIntegerValueField", return_value=SYNTHETIC_EVENT_TAG):
            self.assertIs(listener._tap_callback(None, 10, event, None), event)
        self.assertFalse(listener._is_active)
        listener.on_start.assert_not_called()


class TestIncrementalPreview(IsolatedTestCase):
    def make_engine(self):
        engine = GigaAMEngine(device="cpu")
        engine._is_loaded = True
        engine.model = MagicMock(_dtype=torch.float32, _device=torch.device("cpu"))
        engine.model.forward.return_value = (torch.zeros((1, 5, 10)), torch.tensor([5]))
        engine.model._decode.return_value = ["fragment"]
        return engine

    def test_only_tail_is_rerecognized_after_a_chunk_is_committed(self):
        engine = self.make_engine()
        audio = np.ones(45 * 16000, dtype=np.float32)
        engine.transcribe_preview(audio)
        committed = engine.preview_offset
        self.assertGreater(committed, 0)
        previous_calls = engine.model.forward.call_count
        engine.transcribe_preview(np.concatenate([audio, np.ones(16000, dtype=np.float32)]))
        self.assertEqual(engine.model.forward.call_count - previous_calls, 1)
        self.assertEqual(engine.preview_offset, committed)
        self.assertTrue(all(c.args[0].shape[-1] <= 20 * 16000 for c in engine.model.forward.call_args_list))

    def test_offset_input_matches_full_input_and_discards_old_offset(self):
        full_engine, tail_engine = self.make_engine(), self.make_engine()
        audio = np.ones(45 * 16000, dtype=np.float32)
        self.assertEqual(full_engine.transcribe_preview(audio), tail_engine.transcribe_preview(audio, offset=0))
        offset = tail_engine.preview_offset
        audio = np.concatenate([audio, np.ones(16000, dtype=np.float32)])
        self.assertEqual(full_engine.transcribe_preview(audio), tail_engine.transcribe_preview(audio[offset:], offset=offset))
        calls = tail_engine.model.forward.call_count
        self.assertEqual(tail_engine.transcribe_preview(audio, offset=0), "")
        self.assertEqual(tail_engine.model.forward.call_count, calls)

    def test_cancellation_prevents_remaining_chunks_and_cache_updates(self):
        engine = self.make_engine()
        cancelled = [False]
        def decode(_):
            cancelled[0] = True
            return "stale"
        with patch.object(engine, "_decode_preview_chunk", side_effect=decode) as decoding:
            result = engine.transcribe_preview(np.ones(60 * 16000), cancelled=lambda: cancelled[0])
        self.assertEqual(result, "")
        self.assertEqual(decoding.call_count, 1)
        self.assertEqual(engine.preview_offset, 0)
        self.assertEqual(engine._last_preview_text, "")

    def test_cache_reset_during_inference_cannot_contaminate_new_session(self):
        engine = self.make_engine()
        def decode(_):
            engine.reset_preview_cache()
            return "stale"
        with patch.object(engine, "_decode_preview_chunk", side_effect=decode):
            self.assertEqual(engine.transcribe_preview(np.ones(16000)), "")
        self.assertEqual(engine._last_preview_text, "")
        self.assertEqual(engine.preview_offset, 0)

    def test_final_pass_remains_independent_of_preview_cache(self):
        engine = self.make_engine()
        audio = np.ones(45 * 16000, dtype=np.float32)
        engine.transcribe_preview(audio)
        engine.model.forward.reset_mock()
        engine.model._decode.return_value = ["final"]
        self.assertIn("final", engine.transcribe(audio))
        self.assertGreater(engine.model.forward.call_count, 1)

    def test_snapshot_from_offset_is_exact_and_does_not_modify_recording(self):
        recorder = AudioRecorder()
        recorder._recording = True
        for i in range(10):
            frame = np.full((512, 1), i, dtype=np.float32)
            recorder._callback(frame, 512, None, None)
        expected = recorder.get_stream_snapshot()
        for offset in (0, 256, 512, 5120, 6000):
            np.testing.assert_array_equal(recorder.get_stream_snapshot_from(offset), expected[offset:])
        self.assertTrue(recorder.is_recording)
        self.assertEqual(recorder._sample_count, expected.size)
        recorder.close()


class TestControllerRaces(IsolatedTestCase):
    def controller(self):
        # Instantiate the actual NSObject controller with all native collaborators
        # replaced. No NSApplication, windows, devices or system pasteboard.
        with (patch("giga_dictate.menu_app.VoiceInputHUD") as _hud,
              patch("giga_dictate.menu_app.SoundManager"),
              patch("giga_dictate.menu_app.AudioDucker")):
            controller = MenuAppController.alloc().init()
        controller.hud = MagicMock()
        controller.recorder = MagicMock(is_recording=False)
        controller.engine = MagicMock(is_loaded=True)
        controller.sounds = MagicMock()
        controller.ducker = MagicMock()
        controller.hotkey_listener = MagicMock()
        controller.rebuild_menu = MagicMock()
        controller.update_menu_status = MagicMock()
        controller.update_status_title = MagicMock()
        controller._start_preview_timer = MagicMock()
        controller._stop_preview_timer = MagicMock()
        controller._arm_recovery_expiry = MagicMock()
        return controller

    def test_busy_start_never_restarts_microphone_or_hud(self):
        controller = self.controller()
        controller.session.begin()
        controller.session.state = DictationState.TRANSCRIBING
        controller.start_recording()
        controller.recorder.start.assert_not_called()
        controller.hud.showRecording.assert_not_called()
        controller.sounds.play_cancel.assert_called_once()

    def test_failed_insertion_never_plays_success_and_retains_result(self):
        controller = self.controller()
        controller.session.begin()
        controller.session.state = DictationState.INSERTING
        controller.recovery.remember("recover me")
        controller._insertion_ready(controller.session.generation, InsertionResult("error", "target_changed"))
        controller.sounds.play_success.assert_not_called()
        self.assertEqual(controller.session.state, DictationState.ERROR)
        self.assertEqual(controller.recovery.get(), "recover me")

    def test_old_completion_cannot_hide_new_hud(self):
        controller = self.controller()
        controller.session.begin()
        old = controller.session.generation
        controller.session.finish(old)
        controller.session.begin()
        controller._insertion_ready(old, InsertionResult("event_posted"))
        controller.hud.hideHUD.assert_not_called()
        self.assertEqual(controller.session.state, DictationState.RECORDING)

    def test_old_transcription_cannot_replace_recovery_text(self):
        controller = self.controller()
        controller.session.begin()
        old = controller.session.generation
        controller.session.finish(old)
        controller.session.begin()
        controller.recovery.remember("current")
        controller._transcription_ready(old, "stale", "")
        self.assertEqual(controller.recovery.get(), "current")

    def test_success_uses_async_result_and_cleans_up(self):
        controller = self.controller()
        controller.session.begin()
        controller.session.state = DictationState.TRANSCRIBING
        controller.recorder.is_recording = False
        result = Future()
        controller.inserter = MagicMock()
        controller.inserter.insert_text.return_value = result
        with patch("giga_dictate.menu_app.AppHelper.callAfter", side_effect=lambda f, *args: f(*args)):
            controller._transcription_ready(controller.session.generation, "new text", "")
            self.assertEqual(controller.session.state, DictationState.INSERTING)
            controller.sounds.play_success.assert_not_called()
            result.set_result(InsertionResult("event_posted"))
        self.assertEqual(controller.session.state, DictationState.IDLE)
        self.assertEqual(controller.recovery.get(), "new text")
        controller.hud.hideHUD.assert_called_once()
        controller.sounds.play_success.assert_called_once()

    def test_model_change_during_recording_is_ignored(self):
        controller = self.controller()
        controller.session.begin()
        sender = MagicMock()
        controller.onSelectModel_(sender)
        sender.representedObject.assert_not_called()

    def test_manual_start_synchronizes_hotkey_and_does_not_clear_on_busy_start(self):
        controller = self.controller()
        controller.config.live_hud_preview = False
        controller.recorder.start.return_value = True
        controller.start_recording(manual=True)
        controller.hotkey_listener.set_recording.assert_called_once_with(True)
        controller.start_recording()
        self.assertEqual(controller.recorder.start.call_count, 1)
        self.assertEqual(controller.hud.showRecording.call_count, 1)

    def test_start_is_rejected_while_model_reload_is_pending(self):
        controller = self.controller()
        controller._model_reloading = True
        controller.start_recording()
        controller.recorder.start.assert_not_called()
        self.assertEqual(controller.session.state, DictationState.IDLE)

    def test_shutdown_is_idempotent_and_cleans_up_after_component_failure(self):
        controller = self.controller()
        controller.session.begin()
        controller.recovery.remember("private")
        controller.inserter = MagicMock()
        controller.hud.close.side_effect = RuntimeError("native failure")
        with (patch("giga_dictate.menu_app._WORKER_EXECUTOR"),
              patch("giga_dictate.menu_app._PREVIEW_EXECUTOR")):
            controller._shutdown()
            controller._shutdown()
        self.assertEqual(controller.recovery.get(), "")
        controller.ducker.unduck.assert_called_once()
        controller.recorder.close.assert_called_once()
        controller.hotkey_listener.stop.assert_called_once()

    def test_preview_snapshot_is_not_taken_on_main_thread(self):
        controller = self.controller()
        controller.recorder.is_recording = True
        controller._is_preview_running = False
        controller._preview_future = None
        with patch("giga_dictate.menu_app._PREVIEW_EXECUTOR") as executor:
            controller.onPreviewTimerTick_(None)
        controller.recorder.get_stream_snapshot_from.assert_not_called()
        executor.submit.assert_called_once()
