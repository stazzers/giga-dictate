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
            if owner is not None and owner.active:
                owner.fail("Захват системного звука прерван. Проверьте разрешение записи экрана и звука.")

    _receiver_class = GigaMeetingStreamOutput
    return _receiver_class


class SystemAudioStream:
    def __init__(self, emit, fail):
        self.emit, self.fail = emit, fail
        self.stream = self.receiver = None
        self.active = False
        self.sequence = 0

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
        if not done.wait(15) or values.get("error") or not values.get("content"):
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
        self.receiver = receiver_class().alloc().init()
        self.receiver.owner = self
        self.stream = SC.SCStream.alloc().initWithFilter_configuration_delegate_(content_filter, config, self.receiver)
        ok, error = self.stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self.receiver, SC.SCStreamOutputTypeAudio, None, None)
        if not ok:
            raise ValueError("Не удалось подключить системный аудиопоток")
        before = time.monotonic()
        native_now = CM.CMTimeGetSeconds(CM.CMClockGetTime(CM.CMClockGetHostTimeClock()))
        self.offset = (before + time.monotonic()) / 2 - native_now
        done.clear()
        values.clear()

        def started(error):
            values["error"] = error
            done.set()

        self.active = True
        self.stream.startCaptureWithCompletionHandler_(started)
        if not done.wait(15) or values.get("error"):
            self.stop()
            raise ValueError("Не удалось начать системную запись. Проверьте разрешения macOS.")

    def stop(self):
        self.active = False
        stream, self.stream = self.stream, None
        if stream is not None:
            done = threading.Event()
            stream.stopCaptureWithCompletionHandler_(lambda error: done.set())
            if not done.wait(5):
                raise RuntimeError("Не удалось подтвердить остановку системного аудио")
        if self.receiver is not None:
            self.receiver.owner = None
        self.receiver = None
