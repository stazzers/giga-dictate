"""A hung native audio call must not own the UI thread or process exit."""
import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import numpy as np

from giga_dictate.audio_control import AudioExecutor
from giga_dictate.inserter import InsertionResult
from giga_dictate.session import DictationState
from tests.support import IsolatedTestCase

def make_controller(test):
    from tests.test_stability import TestControllerRaces
    return TestControllerRaces.controller(test)


class AudioResponsiveness(IsolatedTestCase):
    def test_cleanup_watchdog_preserves_full_asr_and_retry_then_next_recording(self):
        controller = make_controller(self)
        controller.session.begin()
        generation = controller.session.generation
        controller.session.state = DictationState.TRANSCRIBING
        controller._captured_audio_generation = generation
        controller.recorder.is_recording = False
        controller.engine._last_preview_text = "Неполный предпросмотр"
        cleanup = Future()
        controller._audio_future = cleanup
        controller._audio_timeout(generation, cleanup)
        self.assertEqual(controller.session.state, DictationState.TRANSCRIBING)
        self.assertEqual(controller.recovery.get(), "")
        first = Future()
        first.set_result(InsertionResult("error", "target_changed"))
        retry = Future()
        retry.set_result(InsertionResult("event_posted"))
        controller.inserter = MagicMock()
        controller.inserter.insert_text.side_effect = [first, retry]
        with patch("giga_dictate.menu_app.AppHelper.callAfter", side_effect=lambda f, *a: f(*a)), \
             patch("giga_dictate.menu_app.AppHelper.callLater", side_effect=self.scheduler.call_later):
            controller._transcription_ready(generation, "Полный результат диктовки", "")
            self.assertEqual(controller.recovery.get(), "Полный результат диктовки")
            self.assertEqual(controller.session.state, DictationState.ERROR)
            controller.onRetryInsertion_(None)
            self.scheduler.run_all()
        self.assertEqual(controller.session.state, DictationState.IDLE)
        self.assertTrue(controller.audio_busy)
        cleanup.set_result(None)
        controller._audio_task_done(generation, cleanup)
        self.assertFalse(controller.audio_busy)
        controller.recorder.start.return_value = True
        controller.start_recording()
        controller.recorder.start.assert_called_once()
        self.assertEqual(controller.session.state, DictationState.RECORDING)

    def test_cancelled_insertion_future_does_not_leave_dictation_busy(self):
        controller = make_controller(self)
        controller.session.begin()
        controller.session.state = DictationState.TRANSCRIBING
        controller.recorder.is_recording = False
        result = Future()
        result.cancel()
        controller.inserter = MagicMock()
        controller.inserter.insert_text.return_value = result
        with patch("giga_dictate.menu_app.AppHelper.callAfter", side_effect=lambda f, *a: f(*a)):
            controller._transcription_ready(controller.session.generation, "Полный текст", "")
        self.assertFalse(controller.session.busy)
        self.assertEqual(controller.recovery.get(), "Полный текст")

    def test_full_recording_reaches_asr_while_native_cleanup_is_blocked(self):
        controller = make_controller(self)
        executor = AudioExecutor()
        controller._audio_io = executor
        self.addCleanup(executor.shutdown)
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        controller.session.begin()
        controller.recorder.is_recording = False
        full_audio = np.full(16000, .1, np.float32)
        controller.recorder.finish_capture.return_value = (full_audio, 1.)
        controller.recorder.close.side_effect = lambda: (entered.set(), release.wait(3))[-1]
        callbacks = []
        with patch("giga_dictate.menu_app._dispatch_audio", side_effect=lambda f, *a: callbacks.append((f, a))):
            controller.stop_and_transcribe()
            self.assertTrue(entered.wait(1))
            self.assertEqual(callbacks[0][0], controller._audio_stopped)
            self.assertIs(callbacks[0][1][1], full_audio)
            controller._audio_timeout(controller.session.generation, controller._audio_future)
            self.assertEqual(controller.session.state, DictationState.TRANSCRIBING)
            release.set()
            controller._audio_future.result(timeout=2)

    def test_blocked_stop_returns_ui_and_watchdog_keeps_preview(self):
        controller = make_controller(self)
        executor = AudioExecutor()
        controller._audio_io = executor
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(executor.shutdown)
        self.addCleanup(release.set)
        controller.session.begin()
        controller.engine._last_preview_text = "Сохранённый предпросмотр"
        controller.recorder.finish_capture.side_effect = lambda: (entered.set(), release.wait(3), ([], 0))[-1]
        callbacks = []
        with patch("giga_dictate.menu_app._dispatch_audio", side_effect=lambda f, *a: callbacks.append((f, a))):
            started = time.monotonic()
            controller.stop_and_transcribe()
            self.assertLess(time.monotonic() - started, .1)
            self.assertTrue(entered.wait(1))
            controller.ducker.unduck.assert_not_called()
            future = controller._audio_future
            controller._audio_timeout(controller.session.generation, future)
            self.assertEqual(controller.session.state, DictationState.ERROR)
            self.assertEqual(controller.recovery.get(), "Сохранённый предпросмотр")
            self.assertTrue(controller.audio_busy)
            controller.start_recording()
            controller.recorder.start.assert_not_called()
            release.set()
            future.result(timeout=2)
        for callback, args in callbacks:
            callback(*args)
        self.assertFalse(controller.audio_busy)

    def test_full_audio_dispatches_before_input_close_and_volume_restore(self):
        controller = make_controller(self)
        order = []
        controller.recorder.finish_capture.side_effect = lambda: (order.append("capture"), ([], 0))[-1]
        controller.recorder.close.side_effect = lambda: order.append("close")
        controller.ducker.unduck.side_effect = lambda: order.append("restore")
        with patch("giga_dictate.menu_app._dispatch_audio", side_effect=lambda *args: order.append("result")):
            controller._stop_audio(1)
        self.assertEqual(order, ["capture", "result", "close", "restore"])

    def test_output_ducks_before_input_starts(self):
        controller = make_controller(self)
        controller.session.begin()
        controller.config.audio_ducking = True
        order = []
        controller.ducker.duck.side_effect = lambda **kwargs: order.append("duck")
        controller.recorder.start.side_effect = lambda: (order.append("start"), True)[-1]
        self.assertTrue(controller._start_audio(1))
        self.assertEqual(order, ["duck", "start"])

    def test_audio_queue_serializes_io_and_shutdown_has_a_deadline(self):
        executor = AudioExecutor()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        order = []
        first = executor.submit(lambda: (entered.set(), release.wait(3), order.append(1)))
        second = executor.submit(lambda: order.append(2))
        self.assertTrue(entered.wait(1))
        started = time.monotonic()
        executor.shutdown(wait=True)
        self.assertLess(time.monotonic() - started, .4)
        self.assertTrue(executor.thread.daemon)
        self.assertFalse(second.done())
        release.set()
        first.result(timeout=2)
        second.result(timeout=2)
        self.assertEqual(order, [1, 2])

    def test_meeting_handoff_closes_dictation_before_capture(self):
        controller = make_controller(self)
        order = []
        controller.recorder.close.side_effect = lambda: order.append("release")
        controller.ducker.unduck.side_effect = lambda: order.append("restore")
        controller.prepare_meeting_audio()
        self.assertEqual(order, ["release", "restore"])

    def test_delayed_native_start_is_released_after_timeout(self):
        controller = make_controller(self)
        controller.session.begin()
        controller.config.audio_ducking = False

        def late_start():
            controller.session.finish(1, "timeout")
            return True

        controller.recorder.start.side_effect = late_start
        self.assertFalse(controller._start_audio(1))
        controller.recorder.close.assert_called_once()

    def test_failed_handoff_does_not_allow_meeting_to_open_microphone(self):
        controller = make_controller(self)
        controller.recorder.close.return_value = False
        with self.assertRaisesRegex(ValueError, "Не удалось освободить"):
            controller.prepare_meeting_audio()
        controller.ducker.unduck.assert_called_once()

    def test_hung_shutdown_does_not_retry_portaudio_on_main_thread(self):
        controller = make_controller(self)
        executor = AudioExecutor()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(executor.shutdown)
        self.addCleanup(release.set)
        executor.submit(lambda: (entered.set(), release.wait(3)))
        self.assertTrue(entered.wait(1))
        controller._audio_io = executor
        with patch("giga_dictate.menu_app._WORKER_EXECUTOR"), patch("giga_dictate.menu_app._PREVIEW_EXECUTOR"):
            started = time.monotonic()
            controller._shutdown()
            self.assertLess(time.monotonic() - started, .5)
        controller.recorder.detach_exit_handler.assert_called_once()
