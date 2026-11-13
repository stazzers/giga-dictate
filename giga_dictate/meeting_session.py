"""One worker owns inference/results; callbacks only enqueue bounded PCM."""
from collections import deque
from datetime import datetime
import threading
import time

from .diarization import CAMPlusEmbedder, Enrollment, OnlineDiarizer, OwnerProfile
from .meeting_audio import FrameNormalizer, MicrophoneStream
from .meeting_models import find_models
from .meeting_types import MeetingResult, MeetingState, Speaker, TranscriptTurn
from .vad import SileroProbability, StreamingVAD


class MeetingSession:
    def __init__(self, engine, profile_path, notify=lambda event: None, *,
                 model_finder=find_models, microphone_factory=MicrophoneStream,
                 system_factory=None, vad_factory=SileroProbability,
                 embedder_factory=CAMPlusEmbedder, writer_factory=None, clock=time.monotonic):
        self.engine, self.notify = engine, notify
        self.profile = OwnerProfile(profile_path)
        self.model_finder, self.microphone_factory = model_finder, microphone_factory
        self.system_factory, self.writer_factory = system_factory, writer_factory
        self.vad_factory, self.embedder_factory = vad_factory, embedder_factory
        self.clock = clock
        self.state = MeetingState.IDLE
        self.generation = 0
        self.result = None
        self.worker = None
        self._capture_lock = threading.Lock()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._closed = threading.Event()
        self._cancel = threading.Event()
        self._frames = deque()
        self._queued = {}
        self.max_queued_seconds = {}
        self._sources = []
        self._failure = ""
        self.started = 0.0
        self.stopped = 0.0

    @property
    def busy(self):
        return self.state in (MeetingState.PREPARING, MeetingState.RECORDING, MeetingState.FINALIZING)

    @property
    def elapsed(self):
        if not self.started:
            return 0.0
        return max(0, (self.stopped or self.clock()) - self.started)

    def _event(self, kind, *, session_id=None, **values):
        self.notify({"session_id": self.generation if session_id is None else session_id,
                     "kind": kind, **values})

    def _state(self, value, session_id=None):
        if session_id is not None and session_id != self.generation:
            return
        self.state = value
        self._event("state", session_id=session_id, state=value)

    def start(self, options):
        options.validate()
        if self.busy:
            return False
        self.generation += 1
        self.options = options
        self.result = None
        self._frames.clear()
        self._queued.clear()
        self.max_queued_seconds.clear()
        self._sources = []
        self._failure = ""
        self.started = self.stopped = 0.0
        self._stop.clear()
        self._closed.clear()
        self._cancel.clear()
        self._state(MeetingState.PREPARING)
        self.worker = threading.Thread(target=self._run, args=(self.generation,), name="GigaMeeting", daemon=True)
        self.worker.start()
        return True

    def _enqueue(self, frame):
        if self._stop.is_set():
            return
        # Non-blocking lock: a capture callback never waits for inference or UI.
        if not self._condition.acquire(blocking=False):
            self.fail("Аудиобуфер недоступен; возможен пропуск аудио. Запись завершена.")
            return
        overflow = False
        try:
            queued = self._queued.get(frame.source, 0.0)
            # Reserve 15 seconds for the VAD/inference phrase (120 s total).
            if queued + frame.duration > 105:
                overflow = True
            else:
                self._queued[frame.source] = queued + frame.duration
                self.max_queued_seconds[frame.source] = max(
                    self.max_queued_seconds.get(frame.source, 0), queued + frame.duration)
                self._frames.append(frame)
                self._condition.notify()
        finally:
            self._condition.release()
        if overflow:
            at = max(0, frame.start - self.started)
            self.fail(f"Очередь переполнена: аудио с {at:.1f} с не обработано; запись завершена.")

    def fail(self, message):
        # Called by capture callbacks; no AppKit, device stop, or file IO here.
        if not self._failure:
            self._failure = message
        self._stop.set()

    def stop(self):
        if not self.busy:
            return
        self._stop.set()

    def cancel(self):
        self._cancel.set()
        self.stop()
        if not self.busy and self.result is not None:
            self.result.clear()
            self.result = None

    def _close_sources(self):
        with self._capture_lock:
            for source in reversed(self._sources):
                try:
                    source.stop()
                except Exception:
                    if not self._failure:
                        self._failure = "Не удалось подтвердить освобождение аудиоустройства"
            self._sources.clear()
        self._closed.set()
        with self._condition:
            self._condition.notify_all()

    def _watch(self):
        # Also handles callback failures while the worker is inside GigaAM.
        self._stop.wait()
        self.stopped = self.clock()
        self._close_sources()

    def _run(self, session_id):
        result = MeetingResult(session_id, datetime.now().astimezone())
        writer = None
        vads, normalizers = {}, {}
        watcher = None
        processing_start = None
        processing_failed = False
        try:
            models = self.model_finder()
            if not self.engine.is_loaded:
                raise ValueError("Дождитесь загрузки GigaAM перед началом встречи")
            embedder = self.embedder_factory(models / "campplus.onnx")
            owner = self.profile.load() if self.options.mode == "room" else None
            diarizer = OnlineDiarizer(embedder, owner=owner)
            enrollment = Enrollment(embedder) if self.options.mode == "enrollment" else None
            result.speakers.update(diarizer.speakers)
            result.speakers["owner"] = Speaker("owner", "Вы")
            sources = ["microphone"] + (["system"] if self.options.mode == "online" else [])
            for name in sources:
                vads[name] = StreamingVAD(name, self.vad_factory(models / "silero_vad.onnx"))
            if self._stop.is_set():
                return
            if self.options.audio_path:
                if self.writer_factory is None:
                    from .meeting_writer import TimelineWriter
                    factory = TimelineWriter
                else:
                    factory = self.writer_factory
                writer = factory(self.options.audio_path, self.options.mode == "online")
                result.audio_path = self.options.audio_path
            self.started = self.clock()
            result.started_at = datetime.now().astimezone()
            normalizers = {name: FrameNormalizer(self.started) for name in sources}
            watcher = threading.Thread(target=self._watch, name="GigaMeetingCaptureStop", daemon=True)
            watcher.start()
            with self._capture_lock:
                if not self._stop.is_set():
                    microphone = self.microphone_factory(self._enqueue, self.fail, self.options.device)
                    self._sources.append(microphone)
                    microphone.start()
                if self.options.mode == "online" and not self._stop.is_set():
                    if self.system_factory is None:
                        from .system_audio import SystemAudioStream
                        factory = SystemAudioStream
                    else:
                        factory = self.system_factory
                    system = factory(self._enqueue, self.fail)
                    self._sources.append(system)
                    system.start()
            if not self._stop.is_set():
                self._state(MeetingState.RECORDING, session_id)

            def recognize(segment):
                nonlocal processing_start
                processing_start = segment.start
                if self._cancel.is_set():
                    return
                if enrollment:
                    enrollment.feed(segment)
                    return
                if self.options.mode == "online" and segment.source == "microphone":
                    pieces = [(segment, "owner")]
                else:
                    pieces = diarizer.split(segment)
                result.speakers.update({key: value for key, value in diarizer.speakers.items()
                                        if key not in result.speakers})
                for piece, speaker in pieces:
                    if len(piece.pcm) < 3200:
                        continue
                    text = self.engine.transcribe(piece.pcm)
                    if text:
                        turn = TranscriptTurn(len(result.turns) + 1, piece.start, piece.end,
                            piece.source, speaker, text,
                            "uncertain" if speaker == "unknown" else "automatic")
                        result.turns.append(turn)
                        self._event("turn", session_id=session_id, turn=turn)

            while not self._cancel.is_set():
                if self._stop.is_set() and self.state != MeetingState.FINALIZING:
                    self._state(MeetingState.FINALIZING, session_id)
                with self._condition:
                    if not self._frames:
                        if self._closed.is_set():
                            break
                        self._condition.wait(.05)
                        continue
                    frame = self._frames.popleft()
                    self._queued[frame.source] -= frame.duration
                processing_start = max(0, frame.start - self.started)
                start, pcm, gap = normalizers[frame.source].convert(frame)
                if gap:
                    result.errors.append(f"Разрыв аудиопотока {frame.source} около {start:.1f} с")
                if writer:
                    writer.feed(frame.source, start, pcm)
                for segment in vads[frame.source].feed(start, pcm):
                    recognize(segment)
                if enrollment and self.elapsed >= 20:
                    self.stop()
            self._stop.set()
            if watcher:
                watcher.join(timeout=35)
            if not self._cancel.is_set():
                for name, normalizer in normalizers.items():
                    start, pcm = normalizer.flush()
                    if len(pcm):
                        if writer:
                            writer.feed(name, start, pcm)
                        for segment in vads[name].feed(start, pcm):
                            recognize(segment)
                    for segment in vads[name].flush():
                        recognize(segment)
                if enrollment and not self._failure:
                    self.profile.save(enrollment.finish())
                    result.enrollment_complete = True
        except ValueError as error:
            processing_failed = True
            # Only controlled validation messages; inference exceptions can contain
            # model inputs and must never be logged or surfaced verbatim.
            self._failure = str(error) if str(error).startswith(("Модели", "Несовмест", "Контроль", "Отсутств", "Голос", "Недостаточно", "Микрофон", "Дождитесь", "Нужно", "Не удалось", "Нет доступного")) else "Ошибка обработки встречи; сохранён частичный результат"
        except Exception:
            processing_failed = True
            self._failure = "Ошибка обработки встречи; сохранён частичный результат"
        finally:
            self._stop.set()
            if watcher:
                watcher.join(timeout=35)
            else:
                self._close_sources()
            self.stopped = self.stopped or self.clock()
            result.duration = self.elapsed
            if self._failure:
                result.errors.append(self._failure)
                if self._frames or (processing_failed and processing_start is not None):
                    first = processing_start if processing_start is not None else result.duration
                    if self._frames:
                        first = min(first, max(0, self._frames[0].start - self.started))
                    result.errors.append(f"Необработанное аудио: с {first:.1f} с до {result.duration:.1f} с")
            if writer:
                try:
                    writer.close(result.duration)
                except Exception:
                    result.errors.append("Ошибка завершения аудиофайла; файл может быть неполным")
            with self._condition:
                self._frames.clear()
                self._queued.clear()
            vads.clear()
            normalizers.clear()
            cancelled = self._cancel.is_set()
            if cancelled:
                result.clear()
            self.result = result
            self._state(MeetingState.ERROR if result.errors else MeetingState.COMPLETED, session_id)
            self._event("complete", session_id=session_id, result=result, cancelled=cancelled)
