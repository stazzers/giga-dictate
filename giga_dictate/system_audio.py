"""ScreenCaptureKit system audio, compatible with macOS 14 (no mic API)."""
import threading
import time

import numpy as np

from .meeting_types import AudioFrame


def decode_sample_buffer(sample):
    """Copy PCM while the retained CMBlockBuffer owns AudioBufferList storage."""
    import CoreAudio as CA
    import CoreMedia as CM
    description = CM.CMSampleBufferGetFormatDescription(sample)
    asbd = CM.CMAudioFormatDescriptionGetStreamBasicDescription(description)
    channels = asbd.mChannelsPerFrame
    floating = bool(asbd.mFormatFlags & 1)
    planar = bool(asbd.mFormatFlags & 32)
    if asbd.mFormatID != int.from_bytes(b"lpcm", "big") or not floating or asbd.mBitsPerChannel != 32:
        raise ValueError("Неподдерживаемый PCM системного звука")
    buffers = CA.AudioBufferList(channels if planar else 1)
    status, _, retained = CM.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sample, None, buffers, 8 + 16 * len(buffers), None, None, 0, None)
    if status:
        raise ValueError("Не удалось прочитать системный аудиобуфер")
    dtype = ">f4" if asbd.mFormatFlags & 2 else "<f4"
    if planar:
        pcm = np.column_stack([np.frombuffer(buffers[i].mData, dtype=dtype)
                               for i in range(channels)])
    else:
        pcm = np.frombuffer(buffers[0].mData, dtype=dtype).reshape(-1, channels).copy()
    # Keep 'retained' alive until after the copy above.
    del retained
    return pcm, int(asbd.mSampleRate), CM.CMTimeGetSeconds(CM.CMSampleBufferGetPresentationTimeStamp(sample))


_receiver_class = None


def receiver_class():
    global _receiver_class
    if _receiver_class is not None:
        return _receiver_class
    import objc
    import ScreenCaptureKit as SC
    from Foundation import NSObject

    class GigaMeetingStreamOutput(NSObject, protocols=[objc.protocolNamed("SCStreamOutput"),
                                                       objc.protocolNamed("SCStreamDelegate")]):
        def stream_didOutputSampleBuffer_ofType_(self, stream, sample, kind):
            owner = self.owner
            if kind != SC.SCStreamOutputTypeAudio or owner is None or not owner.active:
                return
            try:
                pcm, rate, pts = decode_sample_buffer(sample)
                if owner.active:
                    owner.emit(AudioFrame("system", owner.sequence, pts + owner.offset, rate, pcm))
                    owner.sequence += 1
            except Exception:
                if owner.active:
                    owner.fail("Ошибка системного аудиобуфера; запись завершена")

        def stream_didStopWithError_(self, stream, error):
            owner = self.owner
            if owner is not None:
                owner.capture_ended()

    _receiver_class = GigaMeetingStreamOutput
    return _receiver_class


class SystemAudioStream:
    def __init__(self, emit, fail):
        self.emit, self.fail = emit, fail
        self.stream = self.receiver = None
        self.active = False
        self.sequence = 0
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self._started = threading.Event()
        self._released = threading.Event()
        self._released.set()
        self._stop_done = threading.Event()
        self._stop_requested = False

    @property
    def released(self):
        return self._released.is_set()

    def _wait_for_start(self, event):
        deadline = time.monotonic() + 15
        while not self._cancelled.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if event.wait(min(.05, remaining)):
                return True
        return False

    def capture_ended(self):
        was_active = self.active
        self.active = False
        self._released.set()
        self._stop_done.set()
        if was_active:
            self.fail("Захват системного звука прерван. Проверьте разрешение записи экрана и звука.")

    def start(self):
        import CoreMedia as CM
        import ScreenCaptureKit as SC
        from Foundation import NSBundle
        from Quartz import CGPreflightScreenCaptureAccess

        if not CGPreflightScreenCaptureAccess():
            raise ValueError("Нужно разрешение «Запись экрана и системного звука». Откройте настройки и повторите запуск.")
        done, values = threading.Event(), {}

        def content_ready(content, error):
            values.update(content=content, error=error)
            done.set()

        SC.SCShareableContent.getShareableContentWithCompletionHandler_(content_ready)
        received = self._wait_for_start(done)
        if self._cancelled.is_set():
            return
        if not received or values.get("error") or not values.get("content"):
            raise ValueError("Не удалось получить источники системного звука")
        content = values["content"]
        displays = content.displays()
        if not displays:
            raise ValueError("Нет доступного экрана для захвата системного звука")
        own_id = NSBundle.mainBundle().bundleIdentifier()
        excluded = [app for app in content.applications() if app.bundleIdentifier() == own_id]
        content_filter = SC.SCContentFilter.alloc().initWithDisplay_excludingApplications_exceptingWindows_(
            displays[0], excluded, [])
        config = SC.SCStreamConfiguration.alloc().init()
        config.setCapturesAudio_(True)
        config.setExcludesCurrentProcessAudio_(True)
        config.setSampleRate_(16000)
        config.setChannelCount_(2)
        config.setWidth_(2)
        config.setHeight_(2)
        config.setMinimumFrameInterval_(CM.CMTimeMake(1, 1))
        config.setShowsCursor_(False)
        before = time.monotonic()
        native_now = CM.CMTimeGetSeconds(CM.CMClockGetTime(CM.CMClockGetHostTimeClock()))
        self.offset = (before + time.monotonic()) / 2 - native_now
        def started(error):
            if error is not None:
                self.active = False
                self._released.set()
            self._started.set()
            if self._cancelled.is_set():
                self._request_stop()

        with self._lock:
            if self._cancelled.is_set():
                return
            self.receiver = receiver_class().alloc().init()
            self.receiver.owner = self
            self.stream = SC.SCStream.alloc().initWithFilter_configuration_delegate_(
                content_filter, config, self.receiver)
            ok, _ = self.stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self.receiver, SC.SCStreamOutputTypeAudio, None, None)
            if not ok:
                raise ValueError("Не удалось подключить системный аудиопоток")
            if self._cancelled.is_set():
                return
            self.active = True
            self._released.clear()
            try:
                self.stream.startCaptureWithCompletionHandler_(started)
            except Exception:
                # A synchronous exception means no start completion will arrive.
                self._started.set()
                raise
        received = self._wait_for_start(self._started)
        if not self._cancelled.is_set() and (not received or not self.active):
            raise ValueError("Не удалось начать системную запись. Проверьте разрешения macOS.")

    def _request_stop(self):
        with self._lock:
            if self.released or self._stop_requested or not self._started.is_set():
                return
            self._stop_requested = True
            stream = self.stream

        def stopped(error):
            if error is None:
                self._released.set()
            self._stop_done.set()

        try:
            stream.stopCaptureWithCompletionHandler_(stopped)
        except Exception:
            self._stop_done.set()

    def stop(self):
        self._cancelled.set()
        self.active = False
        try:
            # If start is still pending, its completion will issue the stop.
            self._request_stop()
            if not self.released and (not self._stop_done.wait(5) or not self.released):
                raise RuntimeError("Не удалось подтвердить остановку системного аудио")
        finally:
            with self._lock:
                if self.receiver is not None:
                    self.receiver.owner = None
                    self.receiver = None
                if self.released:
                    self.stream = None
