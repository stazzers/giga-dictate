"""Reproduce a paused CoreAudio stream invalidated by another capture session."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from giga_dictate.audio import AudioRecorder
from giga_dictate.meeting_audio import MicrophoneStream


class AudioDevice:
    def __init__(self):
        self.generation = 0
        self.streams = []

    def open(self, callback=None, **kwargs):
        device = self

        class Stream:
            closed = False
            stopped = True
            active = False
            generation = None
            time = 100.0

            def start(self):
                if self.generation is None:
                    device.generation += 1
                    self.generation = device.generation
                self.active, self.stopped = True, False
                # A resumed AudioUnit may report active yet never deliver data
                # once another recorder has taken and released the device.
                if callback and self.generation == device.generation:
                    callback(np.full((512, 1), .1, np.float32), 512,
                             SimpleNamespace(inputBufferAdcTime=100.0), None)

            def stop(self):
                self.active, self.stopped = False, True

            def close(self):
                self.stop()
                self.closed = True

            def abort(self):
                self.stop()

        stream = Stream()
        self.streams.append(stream)
        return stream


class DictationCaptureRegression(unittest.TestCase):
    def test_dictation_meeting_dictation_receives_samples(self):
        device = AudioDevice()
        recorder = AudioRecorder()
        self.addCleanup(recorder.close)
        with patch("giga_dictate.audio.sd.InputStream", side_effect=device.open), \
             patch("sounddevice.query_devices", return_value={"default_samplerate": 48000}):
            self.assertTrue(recorder.start())
            first, _ = recorder.stop()
            frames, errors = [], []
            meeting = MicrophoneStream(frames.append, errors.append)
            meeting.start()
            meeting.stop()
            self.assertEqual(len(frames), 1)
            self.assertEqual(errors, [])
            self.assertTrue(recorder.start())
            after_meeting, duration = recorder.stop()
        self.assertEqual(first.size, 512)
        self.assertEqual(after_meeting.size, 512)
        self.assertEqual(duration, 512 / 16000)

    def test_stop_releases_device_before_another_capture_opens(self):
        device = AudioDevice()
        recorder = AudioRecorder()
        with patch("giga_dictate.audio.sd.InputStream", side_effect=device.open):
            recorder.start()
            recorder.stop()
        self.assertTrue(device.streams[0].closed)
        self.assertIsNone(recorder._stream)

    def test_repeat_dictations_use_fresh_device_and_preserve_preview(self):
        device = AudioDevice()
        recorder = AudioRecorder()
        self.addCleanup(recorder.close)
        with patch("giga_dictate.audio.sd.InputStream", side_effect=device.open):
            for _ in range(5):
                self.assertTrue(recorder.start())
                preview = recorder.get_stream_snapshot_from(0)
                audio, _ = recorder.stop()
                np.testing.assert_array_equal(audio, preview)
                self.assertTrue(device.streams[-1].closed)
        self.assertEqual(len(device.streams), 5)

    def test_start_does_not_reset_an_active_recording(self):
        device = AudioDevice()
        recorder = AudioRecorder()
        self.addCleanup(recorder.close)
        with patch("giga_dictate.audio.sd.InputStream", side_effect=device.open):
            recorder.start()
            self.assertFalse(recorder.start())
            audio, _ = recorder.stop()
        self.assertEqual(audio.size, 512)
        self.assertEqual(len(device.streams), 1)

    def test_stop_error_still_closes_hardware(self):
        device = AudioDevice()
        recorder = AudioRecorder()
        with patch("giga_dictate.audio.sd.InputStream", side_effect=device.open):
            recorder.start()
            stream = device.streams[0]
            original_close = stream.close

            def close():
                # Native close must still run when stop() fails.
                stream.stop = lambda: None
                original_close()

            stream.close = close
            stream.stop = lambda: (_ for _ in ()).throw(RuntimeError("device changed"))
            audio, _ = recorder.stop()
        self.assertTrue(stream.closed)
        self.assertIsNone(recorder._stream)
        self.assertEqual(audio.size, 512)

    def test_close_discards_old_audio_before_next_recording(self):
        device = AudioDevice()
        recorder = AudioRecorder()
        with patch("giga_dictate.audio.sd.InputStream", side_effect=device.open):
            recorder.start()
            recorder.close()
            self.assertFalse(recorder.is_recording)
            self.assertEqual(recorder._sample_count, 0)
            self.assertEqual(recorder._frames, [])
            self.assertTrue(device.streams[0].closed)
            recorder.start()
            audio, _ = recorder.stop()
        self.assertEqual(audio.size, 512)
