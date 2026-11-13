"""Bounded timeline muxer. WAV PCM16 or native AudioToolbox AAC/M4A."""
import ctypes as C
import os
from pathlib import Path
import wave

import numpy as np


class ASBD(C.Structure):
    _fields_ = [("rate", C.c_double), ("format", C.c_uint32), ("flags", C.c_uint32),
                ("bytes_packet", C.c_uint32), ("frames_packet", C.c_uint32),
                ("bytes_frame", C.c_uint32), ("channels", C.c_uint32),
                ("bits", C.c_uint32), ("reserved", C.c_uint32)]


class AudioBuffer(C.Structure):
    _fields_ = [("channels", C.c_uint32), ("size", C.c_uint32), ("data", C.c_void_p)]


class AudioBufferList(C.Structure):
    _fields_ = [("count", C.c_uint32), ("buffers", AudioBuffer * 1)]


def fourcc(text):
    return int.from_bytes(text.encode("ascii"), "big")


class AACFile:
    def __init__(self, path, channels):
        import objc
        from Foundation import NSURL
        self.api = C.CDLL("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")
        self.api.ExtAudioFileCreateWithURL.argtypes = [C.c_void_p, C.c_uint32,
            C.POINTER(ASBD), C.c_void_p, C.c_uint32, C.POINTER(C.c_void_p)]
        self.api.ExtAudioFileSetProperty.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p]
        self.api.ExtAudioFileWrite.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(AudioBufferList)]
        self.api.ExtAudioFileDispose.argtypes = [C.c_void_p]
        self.handle = C.c_void_p()
        self.channels = channels
        url = NSURL.fileURLWithPath_(str(Path(path).absolute()))
        target = ASBD(16000, fourcc("aac "), 0, 0, 1024, 0, channels, 0, 0)
        status = self.api.ExtAudioFileCreateWithURL(objc.pyobjc_id(url), fourcc("m4af"),
            C.byref(target), None, 0, C.byref(self.handle))
        if status:
            raise OSError("Не удалось создать M4A-файл")
        os.chmod(path, 0o600)
        client = ASBD(16000, fourcc("lpcm"), 1 | 8, channels * 4, 1,
                      channels * 4, channels, 32, 0)
        if self.api.ExtAudioFileSetProperty(self.handle, fourcc("cfmt"), C.sizeof(client), C.byref(client)):
            self.close()
            raise OSError("Не удалось настроить AAC-кодировщик")

    def write(self, pcm):
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        buffers = AudioBufferList(1, (AudioBuffer * 1)(AudioBuffer(
            self.channels, pcm.nbytes, pcm.ctypes.data)))
        if self.api.ExtAudioFileWrite(self.handle, len(pcm), C.byref(buffers)):
            raise OSError("Ошибка записи M4A-файла")

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, C.c_void_p()
            if self.api.ExtAudioFileDispose(handle):
                raise OSError("Не удалось завершить M4A-файл")


class TimelineWriter:
    block = 4000

    def __init__(self, path, online):
        self.channels = 2 if online else 1
        self.pending = {}
        self.cursor = 0
        self.latest = 0
        self.path = path
        self.raw = None
        if Path(path).suffix.lower() == ".wav":
            # Keep the exclusive descriptor open: no reopen/truncate race.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            self.raw = os.fdopen(fd, "wb")
            self.file = wave.open(self.raw, "wb")
            self.file.setnchannels(self.channels)
            self.file.setsampwidth(2)
            self.file.setframerate(16000)
            self.wav = True
        else:
            # ExtAudioFileCreateWithURL flags=0 refuses existing files.
            self.file = AACFile(path, self.channels)
            self.wav = False

    def feed(self, source, start, pcm):
        at = round(start * 16000)
        if at < self.cursor:
            overlap = self.cursor - at
            if overlap > 1600:
                raise OSError("Аудиоканал опоздал при сохранении файла; запись завершена")
            pcm, at = pcm[overlap:], self.cursor
        channel = 1 if source == "system" and self.channels == 2 else 0
        pos = 0
        while pos < len(pcm):
            index, offset = divmod(at + pos, self.block)
            part = min(len(pcm) - pos, self.block - offset)
            target = self.pending.setdefault(index, np.zeros((self.block, self.channels), np.float32))
            target[offset:offset + part, channel] = pcm[pos:pos + part]
            pos += part
        self.latest = max(self.latest, at + len(pcm))
        # One second allows delivery skew. Silence from an inactive system source
        # cannot make the other channel accumulate indefinitely.
        self._flush(max(0, self.latest - 16000), final=False)

    def _write(self, data):
        if self.wav:
            self.file.writeframesraw((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
        else:
            self.file.write(data)

    def _flush(self, until, final):
        while self.cursor + self.block <= until or (final and self.cursor < until):
            index = self.cursor // self.block
            size = min(self.block, until - self.cursor)
            data = self.pending.pop(index, None)
            if data is None:
                data = np.zeros((self.block, self.channels), np.float32)
            self._write(data[:size])
            self.cursor += size

    def close(self, duration=None):
        try:
            until = max(self.latest, round((duration or 0) * 16000))
            self._flush(until, final=True)
        finally:
            try:
                self.file.close()
            finally:
                if self.raw:
                    self.raw.close()
                self.pending.clear()
