"""Meeting tests use synthetic PCM and isolated files, never live capture."""
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

from giga_dictate.diarization import Enrollment, OnlineDiarizer, OwnerProfile, unit
from giga_dictate.meeting_audio import FrameNormalizer
from giga_dictate.meeting_session import MeetingSession
from giga_dictate.meeting_types import (AudioFrame, MeetingOptions, MeetingResult,
    MeetingState, Speaker, SpeechSegment, TranscriptTurn)
from giga_dictate.meeting_writer import TimelineWriter
from giga_dictate.vad import StreamingVAD


def vector(index=0):
    result = np.zeros(192, np.float32)
    result[index] = 1
    return result


class Probability:
    def __call__(self, pcm):
        return float(np.max(np.abs(pcm)) > .01)

    def reset(self):
        pass


class MeetingContracts(unittest.TestCase):
    def test_export_sort_overlap_and_manual_override(self):
        result = MeetingResult(1, datetime(2026, 11, 13, 14, 30), duration=120)
        result.speakers = {"a": Speaker("a", "Вы"), "b": Speaker("b", "Собеседник 1")}
        result.turns = [TranscriptTurn(1, 6, 9, "system", "b", "Привет"),
                        TranscriptTurn(2, 3, 8, "microphone", "a", "Как слышно?")]
        result.rename_speaker("b", "Анна")
        result.assign_speaker(1, "a")
        result.turns[0].speaker_id = "b"  # later automated update cannot erase edit
        text = result.render()
        self.assertLess(text.index("Как слышно"), text.index("Привет"))
        self.assertIn("[00:00:06] Вы: Привет", text)
        self.assertTrue(text.startswith("# Стенограмма"))
        self.assertFalse(result.render(False).startswith("#"))
        result.clear()
        self.assertEqual(result.turns, [])

    def test_options_reject_invalid_mode_or_format(self):
        for options in (MeetingOptions("bad"), MeetingOptions(audio_path="x.mp3")):
            with self.assertRaises(ValueError):
                options.validate()

    def test_normalizer_uses_pts_not_arrival(self):
        n = FrameNormalizer(100)
        a = n.convert(AudioFrame("microphone", 0, 102, 16000, np.ones(512, np.float32)))
        b = n.convert(AudioFrame("microphone", 1, 102.032, 16000, np.ones(512, np.float32)))
        self.assertEqual(a[0], 2)
        self.assertAlmostEqual(b[0], 2.032)
        self.assertFalse(b[2])

    def test_gap_is_not_compressed(self):
        n = FrameNormalizer(100)
        n.convert(AudioFrame("system", 0, 100, 16000, np.ones(512, np.float32)))
        start, _, gap = n.convert(AudioFrame("system", 2, 103, 16000, np.ones(512, np.float32)))
        self.assertTrue(gap)
        self.assertEqual(start, 3)

    def test_resampler_is_streaming_and_flushes_tail(self):
        n = FrameNormalizer(100)
        pieces = []
        for i in range(30):
            _, pcm, _ = n.convert(AudioFrame("system", i, 100 + i * .032,
                48000, np.ones((1536, 2), np.float32)))
            pieces.append(pcm)
        _, tail = n.flush()
        pieces.append(tail)
        self.assertEqual(sum(map(len, pieces)), 15360)
        self.assertTrue(np.isfinite(np.concatenate(pieces)).all())

    def test_invalid_pcm_rejected(self):
        with self.assertRaises(ValueError):
            FrameNormalizer(0).convert(AudioFrame("system", 0, 0, 16000, np.array([np.nan])))

    def test_slow_native_clock_drift_does_not_make_audio_gaps(self):
        normalizer = FrameNormalizer(100)
        previous_end = 0
        for i in range(200):
            at = 100 + i * .1001  # 1000 ppm clock error, deliberately exaggerated
            start, pcm, gap = normalizer.convert(AudioFrame("system", i, at, 16000,
                np.full(1600, .1, np.float32)))
            self.assertFalse(gap)
            self.assertAlmostEqual(start, previous_end)
            previous_end = start + len(pcm) / 16000
        self.assertLess(abs(previous_end - 20.02), .002)


class MeetingVAD(unittest.TestCase):
    def test_pause_and_preroll(self):
        vad = StreamingVAD("microphone", Probability())
        results = []
        audio = np.concatenate([np.zeros(8000), np.ones(16000) * .1, np.zeros(16000)]).astype(np.float32)
        for start in range(0, len(audio), 512):
            results.extend(vad.feed(start / 16000, audio[start:start + 512]))
        results.extend(vad.flush())
        self.assertEqual(len(results), 1)
        self.assertLess(results[0].start, .5)
        self.assertGreater(results[0].start, .2)
        self.assertLess(results[0].end, 2.1)

    def test_forced_boundary_and_last_fragment(self):
        vad = StreamingVAD("system", Probability())
        audio = np.full(16000 * 17 + 123, .1, np.float32)
        results = vad.feed(0, audio) + vad.flush()
        self.assertEqual(len(results), 2)
        self.assertLessEqual(len(results[0].pcm), 15 * 16000)
        self.assertEqual(sum(len(x.pcm) for x in results), len(audio))
        self.assertAlmostEqual(results[-1].end, len(audio) / 16000)

    def test_discontinuity_flushes_previous_phrase(self):
        vad = StreamingVAD("system", Probability())
        vad.feed(0, np.full(8000, .1, np.float32))
        result = vad.feed(5, np.full(8000, .1, np.float32)) + vad.flush()
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].end, .5)
        self.assertEqual(result[1].start, 5)

    def test_silence_produces_no_transcript(self):
        vad = StreamingVAD("system", Probability())
        self.assertEqual(vad.feed(0, np.zeros(160000, np.float32)) + vad.flush(), [])


class MeetingDiarization(unittest.TestCase):
    def test_change_without_pause(self):
        embed = lambda pcm: vector(0 if pcm.mean() > 0 else 1)
        diarizer = OnlineDiarizer(embed)
        pcm = np.concatenate([np.ones(16000 * 3) * .1, -np.ones(16000 * 3) * .1]).astype(np.float32)
        pieces = diarizer.split(SpeechSegment("system", 10, 16, pcm))
        self.assertEqual(len(pieces), 2)
        self.assertNotEqual(pieces[0][1], pieces[1][1])
        self.assertAlmostEqual(pieces[0][0].end, pieces[1][0].start)
        self.assertEqual(sum(len(p.pcm) for p, _ in pieces), len(pcm))

    def test_short_reply_remains_unknown(self):
        embed = Mock()
        diarizer = OnlineDiarizer(embed)
        pieces = diarizer.split(SpeechSegment("system", 0, .4, np.ones(6400, np.float32)))
        self.assertEqual(pieces[0][1], "unknown")
        embed.assert_not_called()

    def test_ambiguous_vector_does_not_create_new_speaker(self):
        diarizer = OnlineDiarizer(lambda pcm: vector())
        diarizer.identify(vector())
        ambiguous = vector() * .4 + vector(1) * np.sqrt(1 - .4 ** 2)
        self.assertEqual(diarizer.identify(ambiguous), "unknown")
        self.assertEqual(len(diarizer.centers), 1)

    def test_owner_is_not_updated_from_meeting(self):
        owner = np.stack([vector()] * 3)
        diarizer = OnlineDiarizer(lambda pcm: vector(), owner=owner)
        self.assertEqual(diarizer.identify(vector()), "owner")
        self.assertTrue(np.array_equal(owner, np.stack([vector()] * 3)))
        self.assertNotEqual(diarizer.identify(vector(1)), "owner")

    def test_invalid_voice_vector(self):
        for value in (np.zeros(192), np.ones(10), np.full(192, np.nan)):
            with self.assertRaises(ValueError):
                unit(value)

    def test_enrollment_quality(self):
        enrollment = Enrollment(lambda pcm: vector())
        with self.assertRaises(ValueError):
            enrollment.finish()
        enrollment.feed(SpeechSegment("microphone", 0, 10, np.full(160000, .1, np.float32)))
        self.assertEqual(enrollment.finish().shape, (5, 192))

    def test_enrollment_rejects_clipping_and_other_speakers(self):
        enrollment = Enrollment(lambda pcm: vector())
        enrollment.feed(SpeechSegment("microphone", 0, 10, np.ones(160000, np.float32)))
        with self.assertRaises(ValueError):
            enrollment.finish()
        enrollment.clipped = 0
        enrollment.vectors = [vector(0), vector(1), vector(2), vector(3)]
        with self.assertRaises(ValueError):
            enrollment.finish()

    def test_profile_private_atomic_and_versioned(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = OwnerProfile(Path(tmp) / "private" / "owner.json")
            self.assertIsNone(profile.load())
            profile.save([vector()] * 3)
            self.assertEqual(profile.path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(profile.path.parent.stat().st_mode & 0o777, 0o700)
            np.testing.assert_array_equal(profile.load(), np.stack([vector()] * 3))
            data = json.loads(profile.path.read_text())
            data["model"] = "old-model"
            profile.path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                profile.load()
            profile.delete()
            self.assertIsNone(profile.load())


class MeetingFile(unittest.TestCase):
    def test_wav_channels_gaps_and_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meeting.wav"
            writer = TimelineWriter(str(path), True)
            for i in range(0, 32000, 512):
                size = min(512, 32000 - i)
                writer.feed("microphone", i / 16000, np.full(size, .1, np.float32))
                if i >= 8000:
                    writer.feed("system", i / 16000, np.full(size, .2, np.float32))
            self.assertLessEqual(len(writer.pending), 6)
            writer.close(3)
            with wave.open(str(path)) as handle:
                self.assertEqual(handle.getnchannels(), 2)
                self.assertEqual(handle.getnframes(), 48000)
                data = np.frombuffer(handle.readframes(48000), "<i2").reshape(-1, 2)
            self.assertTrue(np.all(data[:8000, 1] == 0))
            self.assertTrue(np.all(data[32000:] == 0))
            self.assertGreater(data[16000, 1], data[16000, 0])

    def test_existing_file_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meeting.wav"
            path.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                TimelineWriter(str(path), False)
            self.assertEqual(path.read_bytes(), b"original")

    @unittest.skipUnless(os.environ.get("GIGA_NATIVE_AUDIO_TESTS") == "1", "Native AAC encoder requires macOS codec service")
    def test_native_m4a(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meeting.m4a"
            writer = TimelineWriter(str(path), True)
            for i in range(0, 32000, 512):
                size = min(512, 32000 - i)
                writer.feed("microphone", i / 16000, np.full(size, .1, np.float32))
                writer.feed("system", i / 16000, np.full(size, .2, np.float32))
            writer.close(2)
            self.assertIn(b"ftyp", path.read_bytes()[:32])
            import subprocess
            completed = subprocess.run(["/usr/bin/afinfo", str(path)], capture_output=True)
            self.assertEqual(completed.returncode, 0)
            self.assertIn(b"2 ch", completed.stdout)


class Capture:
    def __init__(self, emit, fail, device=None):
        self.emit, self.fail = emit, fail
        self.closed = threading.Event()

    def start(self):
        pass

    def stop(self):
        self.closed.set()


class MeetingLifecycle(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.recording, self.complete = threading.Event(), threading.Event()
        self.events = []
        self.captures = []
        self.engine = Mock(is_loaded=True)
        self.engine.transcribe.return_value = "Проверка встречи"

        def notify(event):
            self.events.append(event)
            if event.get("state") == MeetingState.RECORDING:
                self.recording.set()
            if event["kind"] == "complete":
                self.complete.set()

        def capture(*args):
            source = Capture(*args)
            self.captures.append(source)
            return source

        self.manager = MeetingSession(self.engine, Path(self.temporary.name) / "owner.json", notify,
            model_finder=lambda: Path(self.temporary.name), microphone_factory=capture,
            system_factory=capture, vad_factory=lambda path: Probability(),
            embedder_factory=lambda path: lambda pcm: vector())
        self.addCleanup(self.cleanup_session)

    def cleanup_session(self):
        self.manager.cancel()
        if self.manager.worker:
            self.manager.worker.join(5)

    def start(self, mode="room", **kwargs):
        self.assertTrue(self.manager.start(MeetingOptions(mode, **kwargs)))
        self.assertTrue(self.recording.wait(3))

    def feed(self, source="microphone", seconds=2):
        # One synthetic native buffer also checks arbitrary callback lengths.
        self.manager._enqueue(AudioFrame(source, 0, self.manager.started, 16000,
                                        np.full(int(16000 * seconds), .1, np.float32)))

    def finish(self):
        self.manager.stop()
        self.assertTrue(self.complete.wait(5))
        return self.manager.result

    def test_streams_close_before_inference_finishes_and_tail_once(self):
        self.start()
        entered, release = threading.Event(), threading.Event()

        def transcribe(audio):
            entered.set()
            release.wait(3)
            return "Текст"
        self.engine.transcribe.side_effect = transcribe
        self.feed()
        self.manager.stop()
        self.assertTrue(entered.wait(3))
        self.assertTrue(self.captures[0].closed.is_set())
        self.assertFalse(self.complete.is_set())
        release.set()
        self.assertTrue(self.complete.wait(5))
        self.assertEqual(len(self.manager.result.turns), 1)
        self.engine.transcribe.assert_called_once()

    def test_repeat_start_stop_and_session_ids(self):
        self.start()
        self.assertFalse(self.manager.start(MeetingOptions("room")))
        self.feed()
        first = self.finish()
        self.manager.stop()
        self.recording.clear()
        self.complete.clear()
        self.start()
        self.feed()
        second = self.finish()
        self.assertEqual(second.session_id, first.session_id + 1)
        self.assertEqual(len(second.turns), 1)

    def test_online_sources_remain_separate_and_overlap(self):
        self.start("online")
        self.feed("microphone")
        self.feed("system")
        result = self.finish()
        self.assertEqual(len(self.captures), 2)
        self.assertEqual(len(result.turns), 2)
        by_source = {turn.source: turn for turn in result.turns}
        self.assertEqual(by_source["microphone"].speaker_id, "owner")
        self.assertNotEqual(by_source["system"].speaker_id, "owner")
        self.assertEqual(by_source["system"].start, by_source["microphone"].start)

    def test_no_disk_and_no_network_without_opt_in(self):
        with patch("socket.socket", side_effect=AssertionError("network")), \
             patch("giga_dictate.meeting_writer.TimelineWriter", side_effect=AssertionError("disk")):
            self.start()
            self.feed()
            result = self.finish()
        self.assertTrue(result.turns)
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])

    def test_overflow_is_explicit_and_bounded(self):
        self.start()
        self.feed(seconds=106)
        self.assertTrue(self.complete.wait(5))
        self.assertEqual(self.manager.state, MeetingState.ERROR)
        self.assertIn("переполнена", " ".join(self.manager.result.errors))
        self.assertEqual(len(self.manager._frames), 0)

    def test_source_failure_preserves_partial_result(self):
        self.start()
        self.feed()
        self.captures[0].fail("Устройство отключено")
        self.assertTrue(self.complete.wait(5))
        self.assertTrue(self.manager.result.turns)
        self.assertIn("Устройство отключено", self.manager.result.errors)

    def test_cancel_drops_transcript(self):
        self.start()
        self.feed()
        self.manager.cancel()
        self.assertTrue(self.complete.wait(5))
        self.assertEqual(self.manager.result.turns, [])
        self.assertTrue(self.events[-1]["cancelled"])

    def test_missing_models_do_not_open_microphone(self):
        self.manager.model_finder = Mock(side_effect=ValueError("Модели отсутствуют"))
        self.manager.start(MeetingOptions())
        self.assertTrue(self.complete.wait(3))
        self.assertEqual(self.captures, [])
        self.assertIn("Модели отсутствуют", self.manager.result.errors)

    def test_failed_file_creation_does_not_open_microphone(self):
        self.manager.writer_factory = Mock(side_effect=OSError("disk full"))
        self.manager.start(MeetingOptions("room", audio_path="unused.wav"))
        self.assertTrue(self.complete.wait(3))
        self.assertEqual(self.captures, [])
        self.assertEqual(self.manager.state, MeetingState.ERROR)

    def test_audio_write_failure_releases_streams_and_reports_error(self):
        writer = Mock()
        writer.feed.side_effect = OSError("private path must not be exposed")
        self.manager.writer_factory = Mock(return_value=writer)
        self.start(audio_path=str(Path(self.temporary.name) / "audio.wav"))
        self.feed()
        self.assertTrue(self.complete.wait(5))
        self.assertEqual(self.manager.state, MeetingState.ERROR)
        self.assertNotIn("private path", " ".join(self.manager.result.errors))
        writer.close.assert_called_once()
        self.assertTrue(self.captures[0].closed.is_set())

    def test_enrollment_saves_vectors_without_transcribing(self):
        self.start("enrollment")
        self.feed(seconds=20)
        result = self.finish()
        self.assertTrue(result.enrollment_complete)
        self.assertIsNotNone(self.manager.profile.load())
        self.engine.transcribe.assert_not_called()
        self.assertEqual([p.name for p in Path(self.temporary.name).iterdir()], ["owner.json"])

    def test_failed_system_start_closes_already_open_microphone(self):
        class FailedSystem(Capture):
            def start(self):
                raise ValueError("Не удалось открыть систему")
        self.manager.system_factory = FailedSystem
        self.manager.start(MeetingOptions("online"))
        self.assertTrue(self.complete.wait(5))
        self.assertTrue(self.captures[0].closed.is_set())
        self.assertEqual(self.manager.state, MeetingState.ERROR)

    def test_worker_event_keeps_its_original_generation(self):
        self.start()
        previous = self.manager.generation
        self.finish()
        self.manager.generation += 1
        self.manager._event("turn", session_id=previous, turn=None)
        self.assertEqual(self.events[-1]["session_id"], previous)


class MeetingModelIntegrity(unittest.TestCase):
    def test_corrupt_asset_is_rejected_before_inference(self):
        import hashlib
        from giga_dictate.diarization import MODEL_ID, PREPROCESSING
        from giga_dictate.meeting_models import verify_models
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hashes = {}
            for name in ("campplus.onnx", "silero_vad.onnx", "LICENSE-CAM++.txt", "LICENSE-Silero.txt"):
                content = name.encode()
                (root / name).write_bytes(content)
                hashes[name] = hashlib.sha256(content).hexdigest()
            (root / "manifest.json").write_text(json.dumps({"model": MODEL_ID,
                "preprocessing": PREPROCESSING, "sha256": hashes}))
            self.assertEqual(verify_models(root), root)
            (root / "campplus.onnx").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                verify_models(root)

    def test_explicit_setup_reuses_valid_packaged_assets_without_network(self):
        from giga_dictate.meeting_models import prepare_models
        with patch("giga_dictate.meeting_models.find_models", return_value=Path("prepared")), \
             patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            self.assertEqual(prepare_models(), Path("prepared"))


class CoreMediaDecode(unittest.TestCase):
    def test_interleaved_and_planar_buffers_keep_timestamp(self):
        import CoreMedia as CM
        import CoreAudio as CA
        from CoreFoundation import kCFAllocatorNull
        from giga_dictate.system_audio import decode_sample_buffer
        expected = np.column_stack([np.full(512, .1, np.float32), np.full(512, .2, np.float32)])
        for planar in (False, True):
            with self.subTest(planar=planar):
                flags = 9 | (32 if planar else 0)
                stride = 4 if planar else 8
                description = CA.AudioStreamBasicDescription(48000, int.from_bytes(b"lpcm", "big"),
                    flags, stride, 1, stride, 2, 32, 0)
                status, fmt = CM.CMAudioFormatDescriptionCreate(None, description, 0, None, 0, None, None, None)
                self.assertEqual(status, 0)
                raw = (expected.T if planar else expected).copy().tobytes()
                # Python owns raw; CoreMedia must not free its allocation.
                status, block = CM.CMBlockBufferCreateWithMemoryBlock(None, raw, len(raw),
                    kCFAllocatorNull, None, 0, len(raw), 0, None)
                self.assertEqual(status, 0)
                status, sample = CM.CMAudioSampleBufferCreateReadyWithPacketDescriptions(None,
                    block, fmt, 512, CM.CMTimeMake(123, 10), None, None)
                self.assertEqual(status, 0)
                pcm, rate, pts = decode_sample_buffer(sample)
                self.assertEqual(rate, 48000)
                self.assertEqual(pts, 12.3)
                np.testing.assert_array_equal(pcm, expected)
                del sample, block
                np.testing.assert_array_equal(pcm, expected)  # copied storage


class MeetingMetrics(unittest.TestCase):
    def test_speaker_permutation_and_word_errors(self):
        from scripts.evaluate_meeting import evaluate
        ref = [{"source": "room", "start": 0, "end": 2, "speaker_id": "Анна", "text": "Всё готово"},
               {"source": "room", "start": 2, "end": 4, "speaker_id": "Иван", "text": "Да"}]
        hyp = [{**ref[0], "speaker_id": "speaker-2", "text": "Все готово"},
               {**ref[1], "speaker_id": "speaker-1", "text": "Нет"}]
        scores = evaluate(ref, hyp, collar=0)
        self.assertEqual(scores["non_overlap_der"], 0)
        self.assertAlmostEqual(scores["wer"], 1 / 3)
        hyp[0]["speaker_id"] = "unknown"
        self.assertAlmostEqual(evaluate(ref, hyp, collar=0)["non_overlap_der"], .5)


@unittest.skipUnless(os.environ.get("GIGA_MODEL_TESTS") == "1", "Prepared model smoke tests are opt-in")
class MeetingRealModels(unittest.TestCase):
    def test_prepared_models_on_real_audio(self):
        import soundfile as sf
        from giga_dictate.meeting_models import find_models
        from giga_dictate.diarization import CAMPlusEmbedder
        from giga_dictate.vad import SileroProbability
        models = find_models()
        pcm, rate = sf.read(Path(__file__).resolve().parents[1] / "example.wav", dtype="float32")
        self.assertEqual(rate, 16000)
        with patch("socket.socket", side_effect=AssertionError("network during inference")):
            embedder = CAMPlusEmbedder(models / "campplus.onnx")
            a = embedder(pcm[:24000])
            b = embedder(pcm[:24000])
            self.assertEqual(a.shape, (192,))
            self.assertAlmostEqual(float(a @ b), 1, places=5)
            vad = StreamingVAD("microphone", SileroProbability(models / "silero_vad.onnx"))
            segments = vad.feed(0, pcm) + vad.flush()
            self.assertTrue(segments)
            self.assertLessEqual(segments[-1].end, len(pcm) / rate + .001)


@unittest.skipUnless(os.environ.get("GIGA_UI_TESTS") == "1", "Native windows require GIGA_UI_TESTS=1")
class MeetingNativeUI(unittest.TestCase):
    def setUp(self):
        import AppKit as AK
        from giga_dictate.menu_app import MenuAppController
        from giga_dictate import config
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.patchers = [patch.object(config, "CONFIG_DIR", self.temporary.name),
                         patch.object(config, "CONFIG_PATH", str(Path(self.temporary.name) / "config.json"))]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        AK.NSApplication.sharedApplication()
        self.controller = MenuAppController.alloc().init()
        self.addCleanup(self.controller.recorder.close)
        self.addCleanup(self.controller.hud.close)
        self.addCleanup(self.controller.meeting_ui.close)
        self.ui = self.controller.meeting_ui

    def test_hud_result_edits_and_close(self):
        self.ui.session.state = MeetingState.RECORDING
        self.ui._show_hud()
        self.assertTrue(self.ui.hud.isVisible())
        self.assertFalse(self.ui.hud.canBecomeKeyWindow())
        self.assertFalse(self.ui.hud.canBecomeMainWindow())
        self.ui._hide_hud()
        result = MeetingResult(1, datetime.now(), duration=5)
        result.speakers = {"owner": Speaker("owner", "Вы"), "speaker-1": Speaker("speaker-1", "Собеседник 1")}
        result.turns = [TranscriptTurn(1, 1, 3, "system", "speaker-1", "Привет")]
        self.ui.result = result
        self.ui.onShowResult_(None)
        self.assertIn("Привет", str(self.ui.text.string()))
        self.ui.speaker_popup.selectItemAtIndex_(1)
        self.ui.name_field.setStringValue_("Анна")
        self.ui.onRename_(None)
        self.assertIn("Анна", result.render())
        self.ui.speaker_popup.selectItemAtIndex_(0)
        self.ui.turn_field.setStringValue_("1")
        self.ui.onAssign_(None)
        self.assertIn("Вы: Привет", result.render())
        self.ui.exported = True
        self.ui.onClose_(None)
        self.assertIsNone(self.ui.result)
        self.assertEqual(result.turns, [])

    def test_meeting_blocks_dictation_and_model_changes(self):
        self.ui.session.state = MeetingState.RECORDING
        self.controller.engine._is_loaded = True
        self.controller.rebuild_menu()
        self.assertFalse(self.controller._toggle_menu_item.isEnabled())
        with patch.object(self.controller.recorder, "start") as start:
            self.controller.start_recording()
            start.assert_not_called()
        previous = self.controller.config.model_name
        sender = Mock()
        sender.representedObject.return_value = "v3_ctc"
        self.controller.onSelectModel_(sender)
        self.assertEqual(self.controller.config.model_name, previous)

    def test_stale_completion_cannot_replace_result(self):
        self.ui.session.generation = 2
        result = MeetingResult(1, datetime.now())
        self.ui.deliver({"session_id": 1, "kind": "complete", "result": result, "cancelled": False})
        self.assertIsNone(self.ui.result)

    def test_sleep_only_finishes_active_meeting(self):
        self.ui.session.state = MeetingState.RECORDING
        self.ui.onSleep_(None)
        self.assertTrue(self.ui.session._stop.is_set())
        self.assertIn("сон", self.ui.session._failure)
