"""One worker owns inference/results; callbacks only enqueue bounded PCM."""
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np

from .diarization import CAMPlusEmbedder, OnlineDiarizer, SpeakerCountError, suppress_remote_echo
from .echo_cancel import EchoCanceller
from .meeting_audio import FrameNormalizer, MicrophoneStream
from .meeting_models import find_models
from .meeting_types import MeetingResult, MeetingState, TranscriptTurn
from .vad import SileroProbability, StreamingVAD

MIN_TRANSCRIBE_SAMPLES = 3200


def voiced_pcm(segment):
    pcm = np.asarray(segment.pcm, np.float32)
    mask = segment.speech_mask
    if mask is None or len(mask) != len(pcm):
        return pcm
    return pcm[mask.astype(bool)]


class MeetingSession:
    def __init__(self, engine, notify=lambda event: None, *,
                 model_finder=find_models, microphone_factory=MicrophoneStream,
                 system_factory=None, vad_factory=SileroProbability,
                 embedder_factory=CAMPlusEmbedder, writer_factory=None, clock=time.monotonic):
        self.engine, self.notify = engine, notify
        self.model_finder, self.microphone_factory = model_finder, microphone_factory
        self.system_factory, self.writer_factory = system_factory, writer_factory
        self.vad_factory, self.embedder_factory = vad_factory, embedder_factory
        self.clock = clock
        self.prepare_capture = None
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
        self._releasing = {}
        self._failure = ""
        self.started = 0.0
        self.stopped = 0.0

    @property
    def busy(self):
        return self.state in (MeetingState.PREPARING, MeetingState.RECORDING, MeetingState.FINALIZING)

    @property
    def microphone_unreleased(self):
        return not self._source_released("microphone")

    def _source_released(self, name):
        pending = self._releasing.get(name)
        return pending is None or pending[1].is_set() or getattr(pending[0], "released", False) is True

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
        if self.microphone_unreleased or (options.mode == "online" and not self._source_released("system")):
            raise ValueError("Аудиоустройство не освободилось. Попробуйте снова или перезапустите приложение.")
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

    def _enqueue(self, frame, session_id=None):
        # Only queue bookkeeping shares this lock. Inference and IO run outside it.
        with self._condition:
            if (self._closed.is_set() or self._cancel.is_set() or self._failure
                    or (session_id is not None and session_id != self.generation)):
                return
            queued = self._queued.get(frame.source, 0.0)
            # Reserve 15 seconds for the VAD/inference phrase (120 s total).
            if queued + frame.duration > 105:
                at = max(0, frame.start - self.started)
                self.fail(f"Очередь переполнена: аудио с {at:.1f} с не обработано; запись завершена.", session_id)
                return
            queued += frame.duration
            self._queued[frame.source] = queued
            self.max_queued_seconds[frame.source] = max(self.max_queued_seconds.get(frame.source, 0), queued)
            self._frames.append(frame)
            self._condition.notify()

    def fail(self, message, session_id=None):
        # Called by capture callbacks; no AppKit, device stop, or file IO here.
        with self._condition:
            if self._closed.is_set() or (session_id is not None and session_id != self.generation):
                return
            if not self._failure:
                self._failure = message
            self._stop.set()
            self._condition.notify()

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

    def _start_source(self, name, source):
        with self._capture_lock:
            if self._stop.is_set():
                return
            self._sources.append((name, source))
        try:
            source.start()
        except Exception:
            if not self._stop.is_set():
                raise

    def _close_sources(self, session_id):
        with self._capture_lock:
            sources, self._sources = self._sources, []

        def close_source(source, released):
            try:
                source.stop()
            except Exception:
                return
            released.set()

        workers = []
        for name, source in sources:
            released = threading.Event()
            self._releasing[name] = source, released
            worker = threading.Thread(target=close_source, args=(source, released),
                                      name=f"GigaMeetingStop-{name}", daemon=True)
            workers.append((name, worker))
            worker.start()
        deadline = time.monotonic() + 6
        for name, worker in workers:
            worker.join(max(0, deadline - time.monotonic()))
            if self._source_released(name):
                self._releasing.pop(name, None)
            else:
                self.fail("Не удалось подтвердить освобождение аудиоустройства", session_id)
        with self._condition:
            self._closed.set()
            self._condition.notify_all()

    def _watch(self, session_id):
        # Also handles callback failures while the worker is inside GigaAM.
        self._stop.wait()
        self.stopped = self.clock()
        self._close_sources(session_id)

    def _run(self, session_id):
        result = MeetingResult(session_id, datetime.now().astimezone())
        writer = None
        diarizer = None
        room_diarizer = None
        call_diarizer = None
        online = False
        speaker_evidence = {}
        vads, normalizers = {}, {}
        canceller = None
        watcher = None
        processing_start = None
        processing_failed = False
        try:
            models = self.model_finder()
            if not self.engine.is_loaded:
                raise ValueError("Дождитесь загрузки GigaAM перед началом встречи")
            embedder = self.embedder_factory(models / "campplus.onnx")
            online = self.options.mode == "online"
            count = self.options.speaker_count
            if online:
                room_diarizer = OnlineDiarizer(embedder, id_prefix="room", name_prefix="В комнате")
                call_diarizer = OnlineDiarizer(embedder, id_prefix="call", name_prefix="В созвоне")
                result.speakers.update(room_diarizer.speakers)
                result.speakers.update(call_diarizer.speakers)
            else:
                diarizer = OnlineDiarizer(embedder, speaker_count=count)
                result.speakers.update(diarizer.speakers)
            sources = ["microphone"] + (["system"] if self.options.mode == "online" else [])
            for name in sources:
                vads[name] = StreamingVAD(name, self.vad_factory(models / "silero_vad.onnx"))
            if self.prepare_capture is not None:
                self.prepare_capture()
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
            if online:
                canceller = EchoCanceller()
            watcher = threading.Thread(target=self._watch, args=(session_id,),
                                       name="GigaMeetingCaptureStop", daemon=True)
            watcher.start()

            def emit(frame):
                self._enqueue(frame, session_id)

            def fail(message):
                self.fail(message, session_id)

            if not self._stop.is_set():
                self._start_source("microphone", self.microphone_factory(emit, fail, self.options.device))
            if online and not self._stop.is_set():
                if self.system_factory is None:
                    from .system_audio import SystemAudioStream
                    factory = SystemAudioStream
                else:
                    factory = self.system_factory
                self._start_source("system", factory(emit, fail))
            if not self._stop.is_set():
                self._state(MeetingState.RECORDING, session_id)

            def recognize(segment):
                nonlocal processing_start
                processing_start = segment.start
                if self._cancel.is_set():
                    return
                if online:
                    active = room_diarizer if segment.source == "microphone" else call_diarizer
                    pieces = active.split(segment)
                else:
                    active = diarizer
                    pieces = diarizer.split(segment)
                result.speakers.update({key: value for key, value in active.speakers.items()
                                        if key not in result.speakers})
                for piece, speaker in pieces:
                    audio = voiced_pcm(piece)
                    if len(audio) < MIN_TRANSCRIBE_SAMPLES:
                        continue
                    text = self.engine.transcribe(audio)
                    if text:
                        turn = TranscriptTurn(len(result.turns) + 1, piece.start, piece.end,
                            piece.source, speaker, text,
                            "uncertain" if speaker == "unknown" or speaker.startswith("pending-") else "automatic")
                        result.turns.append(turn)
                        if piece.speaker_evidence is not None:
                            speaker_evidence[turn.id] = piece.speaker_evidence
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
                    if canceller is not None:
                        canceller.note_gap(frame.source)
                if writer:
                    writer.feed(frame.source, start, pcm)
                if canceller is not None:
                    if frame.source == "system":
                        for chunk_start, chunk_pcm in canceller.push_far(start, pcm):
                            for segment in vads["microphone"].feed(chunk_start, chunk_pcm):
                                recognize(segment)
                        for segment in vads["system"].feed(start, pcm):
                            recognize(segment)
                    else:
                        for chunk_start, chunk_pcm in canceller.push_near(start, pcm):
                            for segment in vads["microphone"].feed(chunk_start, chunk_pcm):
                                recognize(segment)
                else:
                    for segment in vads[frame.source].feed(start, pcm):
                        recognize(segment)
            self._stop.set()
            if not self._cancel.is_set():
                for name, normalizer in normalizers.items():
                    start, pcm = normalizer.flush()
                    if len(pcm):
                        if writer:
                            writer.feed(name, start, pcm)
                        if canceller is not None:
                            if name == "system":
                                for chunk_start, chunk_pcm in canceller.push_far(start, pcm):
                                    for segment in vads["microphone"].feed(chunk_start, chunk_pcm):
                                        recognize(segment)
                                for segment in vads["system"].feed(start, pcm):
                                    recognize(segment)
                            else:
                                for chunk_start, chunk_pcm in canceller.push_near(start, pcm):
                                    for segment in vads["microphone"].feed(chunk_start, chunk_pcm):
                                        recognize(segment)
                        else:
                            for segment in vads[name].feed(start, pcm):
                                recognize(segment)
                    if canceller is None:
                        for segment in vads[name].flush():
                            recognize(segment)
                if canceller is not None:
                    for chunk_start, chunk_pcm in canceller.flush():
                        for segment in vads["microphone"].feed(chunk_start, chunk_pcm):
                            recognize(segment)
                    for name in normalizers:
                        for segment in vads[name].flush():
                            recognize(segment)
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
                watcher.join()
            else:
                self._close_sources(session_id)
            self.stopped = self.stopped or self.clock()
            result.duration = self.elapsed
            def restore_preliminary():
                for turn in result.turns:
                    saved = preliminary.get(turn.id)
                    if saved is not None:
                        turn.speaker_id, turn.attribution = saved

            def resolve_all_turns():
                for turn in result.turns:
                    if online:
                        key = turn.speaker_id
                        if key.startswith("call-"):
                            owner = call_diarizer
                        elif key.startswith("room-") or key.startswith("pending-"):
                            owner = room_diarizer
                        else:
                            owner = room_diarizer if turn.source == "microphone" else call_diarizer
                    else:
                        owner = diarizer
                    provisional = turn.speaker_id.startswith("pending-")
                    turn.speaker_id = owner.resolve_speaker(turn.speaker_id)
                    if turn.speaker_id == "unknown":
                        turn.attribution = "uncertain"
                    elif provisional:
                        turn.attribution = "automatic"

            if online and room_diarizer is not None:
                if not self._cancel.is_set():
                    preliminary = {turn.id: (turn.speaker_id, turn.attribution) for turn in result.turns}
                    try:
                        mic_turns = [turn for turn in result.turns if turn.source == "microphone"]
                        sys_turns = [turn for turn in result.turns if turn.source == "system"]
                        mic_evidence = {turn.id: speaker_evidence[turn.id] for turn in mic_turns
                                        if turn.id in speaker_evidence}
                        sys_evidence = {turn.id: speaker_evidence[turn.id] for turn in sys_turns
                                        if turn.id in speaker_evidence}
                        room_diarizer.refine(mic_turns, mic_evidence)
                        call_diarizer.refine(sys_turns, sys_evidence)
                        suppress_remote_echo(result.turns, speaker_evidence, call_diarizer)
                        if count is not None:
                            found = {turn.effective_speaker_id for turn in result.turns
                                     if turn.effective_speaker_id != "unknown"}
                            if count == 1 and any(turn.source == "system" for turn in result.turns):
                                restore_preliminary()
                                result.errors.append(
                                    "Указан один голос, но слышна речь в системном канале; "
                                    "сохранены предварительные назначения")
                            elif len(found) != count:
                                restore_preliminary()
                                result.errors.append(
                                    "Недостаточно независимой речи для указанного количества голосов; "
                                    "сохранены предварительные назначения")
                    except SpeakerCountError as error:
                        restore_preliminary()
                        result.errors.append(str(error))
                    except Exception:
                        restore_preliminary()
                        result.errors.append(
                            "Не удалось уточнить спикеров; сохранены предварительные назначения")
                for owner in (room_diarizer, call_diarizer):
                    result.speakers.update({key: value for key, value in owner.speakers.items()
                                            if key not in result.speakers})
                resolve_all_turns()
                used = {turn.effective_speaker_id for turn in result.turns}
                result.speakers = {key: speaker for key, speaker in result.speakers.items() if key in used}
            elif diarizer is not None:
                if not self._cancel.is_set():
                    preliminary = {turn.id: (turn.speaker_id, turn.attribution) for turn in result.turns}
                    try:
                        diarizer.refine(result.turns, speaker_evidence)
                    except SpeakerCountError as error:
                        restore_preliminary()
                        result.errors.append(str(error))
                    except Exception:
                        restore_preliminary()
                        result.errors.append(
                            "Не удалось уточнить спикеров; сохранены предварительные назначения")
                result.speakers.update({key: value for key, value in diarizer.speakers.items()
                                        if key not in result.speakers})
                resolve_all_turns()
                used = {turn.effective_speaker_id for turn in result.turns}
                result.speakers = {key: speaker for key, speaker in result.speakers.items() if key in used}
            speaker_evidence.clear()
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
                unused_frames, self._frames = self._frames, deque()
                self._queued.clear()
            unused_frames.clear()
            vads.clear()
            normalizers.clear()
            cancelled = self._cancel.is_set()
            if cancelled:
                result.clear()
            self.result = result
            self._state(MeetingState.ERROR if result.errors else MeetingState.COMPLETED, session_id)
            self._event("complete", session_id=session_id, result=result, cancelled=cancelled)
