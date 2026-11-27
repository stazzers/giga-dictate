"""Bounded microphone IPC. A stuck driver can only strand its capture child.

PCM travels through anonymous pipes, never files or application logs. Each
recording owns a fresh child, which must exit before another device is opened.
"""
import json
import logging
import math
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger("giga_dictate.process_audio")
HEADER = struct.Struct("<BI")
PCM, READY, DRAINED, CLOSED, ERROR, WARNING = range(1, 7)
TIMED_PCM, OPENING = 7, 8
TIMING = struct.Struct("<QId")
MAX_PACKET = 65536


def worker_command():
    # The embedded runtime's sys.executable can point into Python.framework.
    # Execute the signed app launcher so helpers keep the same bundle identity.
    launcher = Path(__file__).resolve().parents[2] / "MacOS" / "GigaDictate"
    executable = str(launcher) if launcher.is_file() else sys.executable
    return [executable, "-m", "giga_dictate.audio_worker"]


def spawn_worker(command):
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=0, close_fds=True,
                            cwd=str(Path(__file__).resolve().parents[1]))


class MicrophoneStandby:
    """One launch-ready child owned by the serialized dictation audio executor."""

    def __init__(self):
        self.process = None

    def prepare(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.clear()
        try:
            self.process = spawn_worker([*worker_command(), "--standby"])
        except OSError:
            logger.debug("microphone_standby_unavailable")

    def take(self):
        if self.process is not None and self.process.poll() is not None:
            self.clear()
        process, self.process = self.process, None
        return process

    def clear(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            process.stdin.close()
            process.wait(timeout=.2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=.3)
        finally:
            process.stdout.close()


class ProcessInputStream:
    """InputStream-shaped parent endpoint; native IO lives only in the child."""
    def __init__(self, *, samplerate, channels, dtype, device, callback, blocksize,
                 command=None, start_timeout=3., stop_timeout=.75, timed=False,
                 on_failure=None, standby=None):
        self.config = dict(samplerate=samplerate, channels=channels, dtype=dtype,
                           device=device, blocksize=blocksize, parent_pid=os.getpid())
        if timed:
            self.config["timed"] = True
        self.callback = callback
        self.timed, self.on_failure = timed, on_failure
        self.standby = standby
        self.command = command or worker_command()
        self.start_timeout, self.stop_timeout = start_timeout, stop_timeout
        self.closed, self.active = False, False
        self._process = None
        self._reader = None
        self._ready, self._drained = threading.Event(), threading.Event()
        self._drain_confirmed = False
        self._lifecycle = threading.Lock()
        self._error = False
        self._stop_sent = False
        self._lock = threading.Lock()
        self.first_pcm_at = None
        self._opening_at = self._ready_at = None
        self._blocks = self._warnings = 0

    def start(self):
        requested_at = time.monotonic()
        owns_start = False
        try:
            with self._lifecycle:
                if self.closed or self._process is not None:
                    raise RuntimeError("Microphone worker cannot be reused")
                owns_start = True
                self.active = True
                prepared = self.standby.take() if self.standby is not None else None
                self._process = prepared or spawn_worker(self.command)
                spawned_at = time.monotonic()
                self._reader = threading.Thread(target=self._read, name="GigaMicrophoneIPC", daemon=True)
                self._reader.start()
                self._send(json.dumps(self.config).encode() + b"\n")
            if (not self._ready.wait(self.start_timeout) or self._error
                    or self._stop_sent or self.first_pcm_at is None):
                raise RuntimeError("Microphone worker did not deliver audio")
            opening_at = self._opening_at or spawned_at
            logger.info("microphone_ipc_start standby=%s spawn_ms=%.1f init_ms=%.1f native_ready_ms=%.1f first_pcm_ms=%.1f",
                        prepared is not None, (spawned_at - requested_at) * 1000, (opening_at - spawned_at) * 1000,
                        (self._ready_at - opening_at) * 1000, (self.first_pcm_at - requested_at) * 1000)
        except BaseException:
            if owns_start:
                self.close()
            raise

    def _send(self, data):
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.stdin.write(data)
                self._process.stdin.flush()

    @staticmethod
    def _read_exact(pipe, size):
        if not size:
            return b""
        part = pipe.read(size)
        if not part:
            raise EOFError()
        if len(part) == size:
            return part
        pieces = bytearray(part)
        while len(pieces) < size:
            chunk = pipe.read(size - len(pieces))
            if not chunk:
                raise EOFError()
            pieces.extend(chunk)
        return bytes(pieces)

    def _read(self):
        try:
            # PCM decoding is parent-only. The capture child only forwards bytes.
            import numpy as np

            pipe = self._process.stdout
            while True:
                kind, size = HEADER.unpack(self._read_exact(pipe, HEADER.size))
                if size > MAX_PACKET or kind not in (PCM, TIMED_PCM, OPENING, READY, DRAINED, CLOSED, ERROR, WARNING):
                    raise ValueError("Invalid microphone packet")
                data = self._read_exact(pipe, size)
                if kind in (PCM, TIMED_PCM):
                    received_at = time.monotonic()
                    if (kind == TIMED_PCM) != self.timed:
                        raise ValueError("Unexpected microphone timing format")
                    if self.timed:
                        sequence, rate, timestamp = TIMING.unpack_from(data)
                        if sequence != self._blocks or not 8000 <= rate <= 384000 or not math.isfinite(timestamp):
                            raise ValueError("Invalid microphone timestamp")
                    offset = TIMING.size if self.timed else 0
                    if len(data) <= offset or (len(data) - offset) % 4:
                        raise ValueError("Invalid microphone frame")
                    self._blocks += 1
                    if self.active:
                        pcm = np.frombuffer(data, dtype="<f4", offset=offset).reshape(-1, 1)
                        if self.timed:
                            self.callback(pcm, sequence, rate, timestamp)
                        else:
                            self.callback(pcm, len(pcm), None, None)
                        if self.first_pcm_at is None:
                            self.first_pcm_at = received_at
                    if self._ready_at is not None and self.first_pcm_at is not None:
                        self._ready.set()
                elif kind == READY:
                    self._ready_at = time.monotonic()
                    if self.first_pcm_at is not None:
                        self._ready.set()
                elif kind == OPENING:
                    self._opening_at = time.monotonic()
                elif kind == DRAINED:
                    # FIFO barrier: all preceding capture callbacks are done.
                    self._drain_confirmed = True
                    self._drained.set()
                elif kind == ERROR:
                    self._error = True
                    self._ready.set()
                    if self.on_failure is not None:
                        self.on_failure("Микрофон потерял аудиопоток; запись завершена")
                elif kind == WARNING:
                    self._warnings += 1
                    logger.warning("microphone_input_overflow")
                    if self.on_failure is not None:
                        self.on_failure("Микрофон сообщил о потере аудио; запись завершена")
                else:
                    if not self._stop_sent:
                        self._error = True
                    break
        except Exception:
            if not self._stop_sent:
                self._error = True
        finally:
            if not self._stop_sent and not self._error:
                self._error = True
            if not self._stop_sent and self.on_failure is not None and self._ready_at is not None:
                self.on_failure("Микрофон отключён; запись завершена")
            self._ready.set()
            self._drained.set()

    def finish_capture(self):
        """Drain queued PCM independently of native stop/close in the child."""
        with self._lifecycle:
            self._finish_capture()

    def _finish_capture(self):
        if self._stop_sent:
            return
        self._stop_sent = True
        self._ready.set()
        try:
            self._send(b"STOP\n")
        except (OSError, ValueError):
            pass
        self._drained.wait(.3)
        if not self._drain_confirmed and self.first_pcm_at is not None:
            logger.warning("microphone_drain_timeout")
            if self.on_failure is not None:
                self.on_failure("Не удалось дочитать конец аудиопотока микрофона")
        self.active = False

    def stop(self):
        self.close()

    def abort(self):
        self.close()

    def close(self):
        with self._lifecycle:
            self._close()

    def _close(self):
        if self.closed:
            return
        process = self._process
        if process is None:
            self.active, self.closed = False, True
            return
        self._finish_capture()
        if process is not None:
            try:
                process.wait(timeout=self.stop_timeout)
            except subprocess.TimeoutExpired:
                logger.warning("microphone_worker_stop_timeout; terminating capture child")
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=.3)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=.5)
            finally:
                self.active = False
            # Never overlap a still-live driver process with another session.
            if process.poll() is None:
                raise RuntimeError("Microphone worker has not exited")
            if self._reader is not None:
                self._reader.join(timeout=.3)
            process.stdin.close()
            process.stdout.close()
        self.closed = True
        logger.info("microphone_ipc_end blocks=%d overflow_events=%d failed=%s",
                    self._blocks, self._warnings, self._error)

