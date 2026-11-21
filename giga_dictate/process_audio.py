"""Bounded microphone IPC. A stuck driver can only strand its capture child.

PCM travels through anonymous pipes, never files or application logs. Each
recording owns a fresh child, which must exit before another device is opened.
"""
import json
import logging
import os
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger("giga_dictate.process_audio")
HEADER = struct.Struct("<BI")
PCM, READY, DRAINED, CLOSED, ERROR, WARNING = range(1, 7)
MAX_PACKET = 65536


def worker_command():
    # The embedded runtime's sys.executable can point into Python.framework.
    # Execute the signed app launcher so helpers keep the same bundle identity.
    launcher = Path(__file__).resolve().parents[2] / "MacOS" / "GigaDictate"
    executable = str(launcher) if launcher.is_file() else sys.executable
    return [executable, "-m", "giga_dictate.audio_worker"]


class ProcessInputStream:
    """InputStream-shaped parent endpoint; native IO lives only in the child."""
    def __init__(self, *, samplerate, channels, dtype, device, callback, blocksize,
                 command=None, start_timeout=3., stop_timeout=.75):
        self.config = dict(samplerate=samplerate, channels=channels, dtype=dtype,
                           device=device, blocksize=blocksize, parent_pid=os.getpid())
        self.callback = callback
        self.command = command or worker_command()
        self.start_timeout, self.stop_timeout = start_timeout, stop_timeout
        self.closed, self.active = False, False
        self._process = None
        self._reader = None
        self._ready, self._drained = threading.Event(), threading.Event()
        self._error = False
        self._stop_sent = False
        self._lock = threading.Lock()

    def start(self):
        if self.closed or self._process is not None:
            raise RuntimeError("Microphone worker cannot be reused")
        self.active = True
        try:
            self._process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
                close_fds=True, cwd=str(Path(__file__).resolve().parents[1]))
            self._reader = threading.Thread(target=self._read, name="GigaMicrophoneIPC", daemon=True)
            self._reader.start()
            self._send(json.dumps(self.config).encode() + b"\n")
            if not self._ready.wait(self.start_timeout) or self._error:
                raise RuntimeError("Microphone worker did not start")
        except BaseException:
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
            pipe = self._process.stdout
            while True:
                kind, size = HEADER.unpack(self._read_exact(pipe, HEADER.size))
                if size > MAX_PACKET or kind not in (PCM, READY, DRAINED, CLOSED, ERROR, WARNING):
                    raise ValueError("Invalid microphone packet")
                data = self._read_exact(pipe, size)
                if kind == PCM:
                    if not size or size % 4:
                        raise ValueError("Invalid microphone frame")
                    if self.active:
                        pcm = np.frombuffer(data, dtype="<f4").reshape(-1, 1)
                        self.callback(pcm, len(pcm), None, None)
                elif kind == READY:
                    self._ready.set()
                elif kind == DRAINED:
                    # FIFO barrier: all preceding capture callbacks are done.
                    self._drained.set()
                elif kind == ERROR:
                    self._error = True
                    self._ready.set()
                    self._drained.set()
                elif kind == WARNING:
                    logger.warning("microphone_input_overflow")
                else:
                    if not self._stop_sent:
                        self._error = True
                    break
        except Exception:
            if not self._stop_sent:
                self._error = True
        finally:
            self._ready.set()
            self._drained.set()

    def finish_capture(self):
        """Drain queued PCM independently of native stop/close in the child."""
        if not self._stop_sent:
            self._stop_sent = True
            try:
                self._send(b"STOP\n")
            except (OSError, ValueError):
                pass
        self._drained.wait(.3)
        self.active = False

    def stop(self):
        self.close()

    def abort(self):
        self.close()

    def close(self):
        if self.closed:
            return
        process = self._process
        if process is None:
            self.active, self.closed = False, True
            return
        self.finish_capture()
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

