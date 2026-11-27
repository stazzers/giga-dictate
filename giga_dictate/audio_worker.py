"""Private capture child. stdout is a framed PCM pipe, not a log stream."""
import json
import os
import select
import sys
import threading
import time
from queue import Full, Queue

from .process_audio import (
    CLOSED,
    DRAINED,
    ERROR,
    HEADER,
    MAX_PACKET,
    OPENING,
    PCM,
    READY,
    TIMED_PCM,
    TIMING,
    WARNING,
)


def main():
    if sys.argv[1:] == ["--standby"]:
        # Cache the interpreter and Python-side imports. PortAudio is initialized
        # only after the parent explicitly requests a recording.
        import cffi  # noqa: F401

    control = sys.stdin.buffer.raw
    config_line = control.readline(4096)
    if not config_line:
        return
    # No atexit Pa_Terminate: the parent enforces this process's lifetime.
    queue = Queue(maxsize=64)
    gate = threading.Lock()
    accepting, failed = threading.Event(), threading.Event()
    stopping, finished, warned = threading.Event(), threading.Event(), threading.Event()
    stream = None
    timed = False
    sequence = 0

    def write():
        try:
            while True:
                item = queue.get()
                if item is None:
                    return
                kind, data = item
                packet = HEADER.pack(kind, len(data)) + data
                while packet:
                    packet = packet[os.write(1, packet):]
        except OSError:
            failed.set()

    writer = threading.Thread(target=write, name="GigaCapturePipe", daemon=True)
    writer.start()

    def send(kind, data=b""):
        queue.put((kind, data), timeout=.25)

    def callback(data, frames, timing, status):
        nonlocal sequence
        if not accepting.is_set():
            return
        with gate:
            if not accepting.is_set():
                return
            if status:
                if not warned.is_set():
                    warned.set()
                    try:
                        queue.put_nowait((WARNING, b""))
                    except Full:
                        failed.set()
                        return
            raw = bytes(data)
            kind = PCM
            if timed:
                stamp = timing.inputBufferAdcTime + clock_offset
                raw = TIMING.pack(sequence, native_rate, stamp) + raw
                kind = TIMED_PCM
            if len(raw) > MAX_PACKET:
                failed.set()
                return
            try:
                queue.put_nowait((kind, raw))
                sequence += 1
            except Full:
                failed.set()

    try:
        config = json.loads(config_line)
        parent_pid = config.pop("parent_pid", os.getppid())

        def lifetime():
            # Also covers app exit while its daemon cleanup thread is waiting.
            # Never leave an orphan AudioUnit holding the microphone.
            while not stopping.wait(.05):
                if os.getppid() != parent_pid:
                    os._exit(0)
            if not finished.wait(.6):
                os._exit(0)

        threading.Thread(target=lifetime, name="GigaCaptureLifetime", daemon=True).start()
        timed = config.pop("timed", False)
        if (config["channels"] != 1 or config["dtype"] != "float32" or
                (config["samplerate"], config["blocksize"]) != ((None, None) if timed else (16000, 512))):
            raise ValueError("Unsupported capture configuration")

        import sounddevice as sd

        send(OPENING)
        if timed:
            native_rate = int(sd.query_devices(config["device"], "input")["default_samplerate"])
            if not 8000 <= native_rate <= 384000:
                raise ValueError("Unsupported device sample rate")
            config.update(samplerate=native_rate, blocksize=max(256, int(native_rate * .032)))

        def ended():
            if accepting.is_set():
                failed.set()

        stream = sd.RawInputStream(callback=callback, finished_callback=ended, **config)
        if timed:
            before = time.monotonic()
            native_now = stream.time
            clock_offset = (before + time.monotonic()) / 2 - native_now
        accepting.set()
        stream.start()
        send(READY)
        while not failed.is_set():
            if select.select([control], [], [], .05)[0]:
                if control.readline(32) != b"STOP\n":
                    failed.set()
                break
        with gate:
            accepting.clear()
        stopping.set()
        if failed.is_set():
            send(ERROR)
        # The acknowledgement precedes the native call which may deadlock.
        send(DRAINED)
        stream.abort()
        stream.close()
        stream = None
        send(CLOSED)
        queue.put(None, timeout=.25)
        writer.join(timeout=.3)
    except BaseException:
        # Never send exception details, model inputs or PCM to stderr/logs.
        try:
            send(ERROR)
            send(DRAINED)
            queue.put(None, timeout=.25)
            writer.join(timeout=.3)
        except (Full, OSError):
            pass
    finally:
        finished.set()
        # This releases all CoreAudio handles, including a partially opened
        # AudioUnit. A native hang above is dealt with by the parent's deadline.
        os._exit(0)


if __name__ == "__main__":
    main()
