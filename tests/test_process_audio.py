"""Real pipe/process lifetime regressions; the worker never opens a microphone."""
import time
import unittest

import numpy as np

from giga_dictate.audio import AudioRecorder
from giga_dictate.process_audio import ProcessInputStream, worker_command


WORKER = r'''
import json, signal, struct, sys, time
mode = sys.argv[1]
json.loads(sys.stdin.buffer.readline())
def send(kind, data=b''):
    sys.stdout.buffer.write(struct.pack('<BI', kind, len(data)) + data)
    sys.stdout.buffer.flush()
if mode == 'start_hang':
    time.sleep(30)
if mode == 'early_exit':
    raise SystemExit(1)
if mode == 'malformed':
    sys.stdout.buffer.write(struct.pack('<BI', 1, 10000000))
    sys.stdout.buffer.flush()
    time.sleep(30)
if mode == 'unkillable_stop':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
send(2)
send(1, struct.pack('<f', .1) * 512)
sys.stdin.buffer.readline()
# A final callback is in flight when STOP is received. It must be captured
# before the DRAINED FIFO marker, even if native shutdown never returns.
send(1, struct.pack('<f', .2) * 256)
send(3)
if mode in ('stop_hang', 'unkillable_stop'):
    time.sleep(30)
send(4)
'''

NATIVE_WORKER_WITH_FAKE_DEVICE = r'''
import sys, types, time
import numpy as np
class Stream:
    def __init__(self, callback, **kwargs):
        self.callback = callback
        if sys.argv[1] == 'open_hang':
            time.sleep(30)
    def start(self):
        self.callback(np.full((512, 1), .1, np.float32), 512, None, None)
    def abort(self):
        time.sleep(30)
    def close(self):
        pass
sys.modules['sounddevice'] = types.SimpleNamespace(InputStream=Stream)
from giga_dictate.audio_worker import main
main()
'''


class ProcessCapture(unittest.TestCase):
    def stream(self, mode, callback=lambda *args: None):
        stream = ProcessInputStream(samplerate=16000, channels=1, dtype="float32",
            device=None, callback=callback, blocksize=512,
            command=[worker_command()[0], "-u", "-c", WORKER, mode],
            start_timeout=.6, stop_timeout=.1)
        self.addCleanup(stream.close)
        return stream

    def test_fifo_tail_is_delivered_before_native_shutdown_finishes(self):
        blocks = []
        stream = self.stream("stop_hang", lambda pcm, *args: blocks.append(pcm.copy()))
        stream.start()
        before = time.monotonic()
        stream.finish_capture()
        self.assertLess(time.monotonic() - before, .5)
        np.testing.assert_array_equal(np.concatenate(blocks).ravel(),
                                     np.r_[np.full(512, .1, np.float32), np.full(256, .2, np.float32)])
        self.assertIsNone(stream._process.poll())
        stream.close()
        self.assertIsNotNone(stream._process.poll())
        self.assertTrue(stream.closed)
        self.assertFalse(stream._reader.is_alive())

    def test_deadlocked_child_is_reaped_even_if_sigterm_is_ignored(self):
        stream = self.stream("unkillable_stop")
        stream.start()
        before = time.monotonic()
        stream.close()
        self.assertLess(time.monotonic() - before, 1.5)
        self.assertIsNotNone(stream._process.poll())
        stream.close()  # idempotent, with no second native stop

    def test_startup_hang_or_bad_protocol_leaves_no_capture_process(self):
        for mode in ("start_hang", "early_exit", "malformed"):
            with self.subTest(mode=mode):
                stream = self.stream(mode)
                with self.assertRaises(RuntimeError):
                    stream.start()
                self.assertTrue(stream.closed)
                self.assertIsNotNone(stream._process.poll())

    def test_recording_result_is_detached_and_next_dictation_gets_fresh_child(self):
        streams = []

        def factory(**kwargs):
            mode = "stop_hang" if not streams else "normal"
            stream = self.stream(mode, kwargs["callback"])
            streams.append(stream)
            return stream

        recorder = AudioRecorder(stream_factory=factory)
        self.addCleanup(recorder.close)
        for _ in range(3):
            self.assertTrue(recorder.start())
            audio, duration = recorder.finish_capture()
            self.assertEqual(len(audio), 768)
            self.assertAlmostEqual(duration, 768 / 16000)
            # The complete audio is available while child native close is hung.
            self.assertFalse(recorder.is_recording)
            self.assertTrue(recorder.close())
            self.assertIsNone(recorder._stream)
            self.assertIsNotNone(streams[-1]._process.poll())
        self.assertEqual(len({s._process.pid for s in streams}), 3)

    def test_repeated_stops_release_all_processes_and_reader_threads(self):
        streams = []
        for i in range(20):
            stream = self.stream("stop_hang" if i % 5 == 0 else "normal")
            streams.append(stream)
            stream.start()
            stream.close()
            self.assertFalse(stream.active)
        self.assertTrue(all(s.closed and s._process.poll() is not None for s in streams))
        self.assertFalse(any(s._reader.is_alive() for s in streams))

    def test_actual_worker_self_exits_when_native_abort_deadlocks(self):
        stream = self.stream("normal")
        stream.command = [worker_command()[0], "-u", "-c", NATIVE_WORKER_WITH_FAKE_DEVICE, "stop_hang"]
        stream.start_timeout, stream.stop_timeout = 2., 2.
        stream.start()
        before = time.monotonic()
        stream.close()
        self.assertLess(time.monotonic() - before, 1.5)
        self.assertEqual(stream._process.returncode, 0)  # child lifetime watchdog, not parent SIGKILL

    def test_actual_worker_exits_if_owner_dies_during_native_open(self):
        stream = self.stream("normal")
        stream.command = [worker_command()[0], "-u", "-c", NATIVE_WORKER_WITH_FAKE_DEVICE, "open_hang"]
        stream.config["parent_pid"] = -1  # simulate a dead owner, without leaving a real orphan
        stream.start_timeout = 2.
        with self.assertRaises(RuntimeError):
            stream.start()
        self.assertEqual(stream._process.returncode, 0)

    def test_actual_module_entry_rejects_config_without_opening_device(self):
        stream = self.stream("normal")
        stream.command = worker_command()
        stream.config["channels"] = 2  # rejected before importing sounddevice
        stream.start_timeout = 2.
        with self.assertRaises(RuntimeError):
            stream.start()
        self.assertEqual(stream._process.returncode, 0)

