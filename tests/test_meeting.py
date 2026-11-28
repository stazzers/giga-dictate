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

from giga_dictate.diarization import OnlineDiarizer, unit
from giga_dictate.meeting_audio import FrameNormalizer
from giga_dictate.meeting_session import MIN_TRANSCRIBE_SAMPLES, MeetingSession, voiced_pcm
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
        result.speakers = {"a": Speaker("a", "Спикер 1"), "b": Speaker("b", "Спикер 2")}
        result.turns = [TranscriptTurn(1, 6, 9, "system", "b", "Привет"),
                        TranscriptTurn(2, 3, 8, "microphone", "a", "Как слышно?")]
        result.rename_speaker("b", "Анна")
        result.assign_speaker(1, "a")
        result.turns[0].speaker_id = "b"  # later automated update cannot erase edit
        text = result.render()
        self.assertLess(text.index("Как слышно"), text.index("Привет"))
        self.assertIn("[00:00:06] Спикер 1: Привет", text)
        self.assertTrue(text.startswith("# Стенограмма"))
        self.assertFalse(result.render(False).startswith("#"))
        result.clear()
        self.assertEqual(result.turns, [])

    def test_options_reject_invalid_mode_or_format(self):
        for options in (MeetingOptions("bad"), MeetingOptions("enrollment"), MeetingOptions(audio_path="x.mp3")):
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

    def test_voiced_pcm_helpers(self):
        spoken = np.full(16000, .1, np.float32)
        silent = np.zeros(32000, np.float32)
        pcm = np.concatenate((spoken, silent))
        mask = np.concatenate((np.ones(16000, bool), np.zeros(32000, bool)))
        segment = SpeechSegment("microphone", 0, len(pcm) / 16000, pcm, mask)
        voiced = voiced_pcm(segment)
        self.assertEqual(len(voiced), 16000)
        self.assertGreater(np.max(np.abs(voiced)), .01)
        short_mask = np.zeros(len(pcm), bool)
        short_mask[:1600] = True
        short = voiced_pcm(SpeechSegment("microphone", 0, 1, pcm, short_mask))
        self.assertLess(len(short), MIN_TRANSCRIBE_SAMPLES)
        bare = SpeechSegment("microphone", 0, 1, spoken, None)
        np.testing.assert_array_equal(voiced_pcm(bare), spoken)

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
    def test_numbered_speakers_have_no_self_role(self):
        for offset in (0, 1):
            with self.subTest(offset=offset):
                diarizer = OnlineDiarizer(lambda pcm: vector(), speaker_offset=offset)
                first = diarizer.identify(vector())
                for _ in range(3):
                    second = diarizer.identify(vector(1))
                self.assertEqual(diarizer.speakers[first].name, f"Спикер {1 + offset}")
                self.assertEqual(diarizer.speakers[second].name, f"Спикер {2 + offset}")
                self.assertNotIn("Вы", [speaker.name for speaker in diarizer.speakers.values()])

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


    def test_invalid_voice_vector(self):
        for value in (np.zeros(192), np.ones(10), np.full(192, np.nan)):
            with self.assertRaises(ValueError):
                unit(value)


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

        self.manager = MeetingSession(self.engine, notify=notify,
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

    def test_system_stops_while_microphone_stop_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        class SlowMicrophone(Capture):
            def stop(self):
                entered.set()
                release.wait(3)
                super().stop()

        system = Capture(None, None)
        self.manager.microphone_factory = SlowMicrophone
        self.manager.system_factory = lambda *args: system
        self.start("online")
        self.manager.stop()
        self.assertTrue(entered.wait(1))
        self.assertTrue(system.closed.wait(1))
        self.assertFalse(self.complete.is_set())
        release.set()
        self.assertTrue(self.complete.wait(3))
        self.assertFalse(self.manager.microphone_unreleased)

    def test_stop_interrupts_preparing_source_and_rejects_its_late_frames(self):
        entered = threading.Event()

        class PreparingSystem(Capture):
            def start(self):
                entered.set()
                self.closed.wait(3)

        self.manager.system_factory = PreparingSystem
        self.assertTrue(self.manager.start(MeetingOptions("online")))
        self.assertTrue(entered.wait(2))
        old_microphone = self.captures[0]
        self.manager.stop()
        self.assertTrue(self.complete.wait(2))
        self.assertTrue(old_microphone.closed.is_set())
        self.complete.clear()
        self.start()
        old_microphone.emit(AudioFrame("microphone", 0, self.manager.started, 16000,
                                       np.full(32000, .1, np.float32)))
        old_microphone.fail("Устаревший сбой")
        result = self.finish()
        self.assertFalse(result.turns)
        self.assertFalse(result.errors)

    def test_queue_contention_does_not_cancel_capture(self):
        self.start()
        entered = threading.Event()

        def feed():
            entered.set()
            self.feed()

        thread = threading.Thread(target=feed)
        with self.manager._condition:
            thread.start()
            self.assertTrue(entered.wait(1))
        thread.join(1)
        self.assertFalse(thread.is_alive())
        result = self.finish()
        self.assertTrue(result.turns)
        self.assertFalse(result.errors)

    def test_online_sources_remain_separate_and_overlap(self):
        self.manager.embedder_factory = lambda path: lambda pcm: vector(0 if pcm.mean() > 0 else 1)
        self.start("online")
        self.feed("microphone")
        self.manager._enqueue(AudioFrame("system", 0, self.manager.started, 16000,
                                         np.full(int(16000 * 2), -.2, np.float32)))
        result = self.finish()
        self.assertEqual(len(self.captures), 2)
        self.assertEqual(len(result.turns), 2)
        by_source = {turn.source: turn for turn in result.turns}
        self.assertEqual(by_source["microphone"].speaker_id, "room-1")
        self.assertEqual(by_source["system"].speaker_id, "call-1")
        self.assertEqual(result.speakers["room-1"].name, "В комнате 1")
        self.assertEqual(result.speakers["call-1"].name, "В созвоне 1")
        self.assertNotIn("Вы", [speaker.name for speaker in result.speakers.values()])
        self.assertEqual(by_source["system"].start, by_source["microphone"].start)

    def test_online_echo_cancel_keeps_raw_recording_and_mic_turn(self):
        self.manager.embedder_factory = lambda path: lambda pcm: vector(0 if pcm.mean() > 0 else 1)
        path = Path(self.temporary.name) / "online.wav"
        self.start("online", audio_path=str(path))
        for i in range(0, 32000, 512):
            size = min(512, 32000 - i)
            pcm = np.full(size, .1, np.float32)
            seq = (i // 512) * 2
            self.manager._enqueue(AudioFrame("microphone", seq, self.manager.started, 16000, pcm))
            self.manager._enqueue(AudioFrame("system", seq + 1, self.manager.started, 16000,
                                             np.full(size, -.2, np.float32)))
        result = self.finish()
        with wave.open(str(path)) as handle:
            data = np.frombuffer(handle.readframes(handle.getnframes()), "<i2").reshape(-1, 2)
        self.assertGreater(int(np.max(np.abs(data[:, 0]))), 2500)
        mic_turns = [turn for turn in result.turns if turn.source == "microphone"]
        self.assertTrue(mic_turns)

    def test_no_disk_and_no_network_without_opt_in(self):
        with patch("socket.socket", side_effect=AssertionError("network")), \
             patch("giga_dictate.meeting_writer.TimelineWriter", side_effect=AssertionError("disk")):
            self.start()
            self.feed()
            result = self.finish()
        self.assertTrue(result.turns)
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])

    def test_room_speakers_start_at_one_and_remain_editable(self):
        self.start("room")
        self.feed()
        result = self.finish()
        turn = result.turns[0]
        self.assertEqual(turn.speaker_id, "speaker-1")
        self.assertEqual(result.speakers[turn.speaker_id].name, "Спикер 1")
        self.assertNotIn("Вы", [speaker.name for speaker in result.speakers.values()])
        result.rename_speaker(turn.speaker_id, "Анна")
        self.assertIn("Анна: Проверка встречи", result.render())

    def test_finalization_recovers_early_uncertain_turns_without_running_asr_twice(self):
        early = vector() * .4 + vector(2) * np.sqrt(1 - .4 ** 2)
        later = vector() * .6 + vector(2) * .8

        def embed(pcm):
            mean = float(pcm.mean())
            return vector() if mean < .15 else early if mean < .25 else later if mean < .35 else vector(1)

        self.manager.embedder_factory = lambda path: embed
        silence = np.zeros(12800, np.float32)
        pcm = np.concatenate([part for level in (.1, .2, .3, .4)
                              for part in (np.full(40000, level, np.float32), silence)])
        self.start("room")
        original = OnlineDiarizer.refine
        before = []

        def refine(diarizer, turns, evidence):
            before.extend(turn.speaker_id for turn in turns)
            return original(diarizer, turns, evidence)

        with patch.object(OnlineDiarizer, "refine", autospec=True, side_effect=refine):
            self.manager._enqueue(AudioFrame("microphone", 0, self.manager.started, 16000, pcm))
            result = self.finish()
        self.assertIn("unknown", before)
        self.assertNotIn("unknown", [turn.speaker_id for turn in result.turns])
        self.assertEqual({turn.speaker_id for turn in result.turns}, {"speaker-1", "speaker-2"})
        self.assertTrue(all(turn.speaker_id == "speaker-1" for turn in result.turns if turn.start < 9.5))
        self.assertEqual(self.engine.transcribe.call_count, len(result.turns))
        self.assertEqual(result.errors, [])
        self.assertNotIn("Спикер не определён:", result.render())

    def test_refinement_failure_still_completes_with_partial_transcript(self):
        self.start("room")
        with patch.object(OnlineDiarizer, "refine", side_effect=RuntimeError("private input")):
            self.feed()
            result = self.finish()
        self.assertTrue(result.turns)
        self.assertTrue(any("Не удалось уточнить" in error for error in result.errors))
        self.assertNotIn("private input", " ".join(result.errors))

    def test_final_pass_registers_new_short_reply_speakers_in_result(self):
        self.manager.embedder_factory = lambda path: lambda pcm: vector(int(round(float(pcm.mean()) * 10)) - 1)
        levels = [.1, .2, .3, .1, .3, .2, .3, .1, .3]
        pcm = np.concatenate([p for i, level in enumerate(levels) for p in
                              (np.full(48000 if i < 2 else 12800, level, np.float32),
                               np.zeros(12800, np.float32))])
        self.start("room")
        self.manager._enqueue(AudioFrame("microphone", 0, self.manager.started, 16000, pcm))
        result = self.finish()
        self.assertEqual({t.speaker_id for t in result.turns}, {"speaker-1", "speaker-2", "speaker-3"})
        self.assertTrue(all(t.speaker_id in result.speakers for t in result.turns))
        self.assertEqual(self.engine.transcribe.call_count, len(result.turns))
        self.assertEqual(result.errors, [])

    def test_exact_online_count_spans_room_and_call(self):
        self.manager.embedder_factory = lambda path: lambda pcm: vector(
            2 if pcm.mean() > .15 else (0 if pcm.mean() > 0 else 1))
        self.start("online", speaker_count=3)
        pcm = np.concatenate([p for level in (.1, -.1, .1, -.1) for p in
                              (np.full(32000, level, np.float32), np.zeros(12800, np.float32))])
        self.manager._enqueue(AudioFrame("system", 0, self.manager.started, 16000, pcm))
        self.manager._enqueue(AudioFrame("microphone", 0, self.manager.started, 16000,
                                         np.full(int(16000 * 2), .2, np.float32)))
        result = self.finish()
        self.assertEqual({t.speaker_id for t in result.turns}, {"room-1", "call-1", "call-2"})
        self.assertEqual(result.errors, [])

    def test_impossible_count_returns_controlled_warning_and_preliminary_text(self):
        self.start("room", speaker_count=6)
        self.feed()
        result = self.finish()
        self.assertTrue(result.turns)
        self.assertEqual({t.speaker_id for t in result.turns}, {"speaker-1"})
        self.assertEqual(set(result.speakers), {"speaker-1"})
        self.assertTrue(any("Недостаточно независимой речи" in error for error in result.errors))

    def test_one_online_voice_rejects_system_speech_but_retains_it(self):
        self.manager.embedder_factory = lambda path: lambda pcm: vector(0 if pcm.mean() > 0 else 1)
        self.start("online", speaker_count=1)
        self.manager._enqueue(AudioFrame("system", 0, self.manager.started, 16000,
                                         np.full(int(16000 * 2), -.2, np.float32)))
        self.feed("microphone")
        result = self.finish()
        self.assertEqual(len(result.turns), 2)
        self.assertTrue(any("системном канале" in error for error in result.errors))

    def test_partial_mutation_by_failed_refiner_is_rolled_back(self):
        def broken(diarizer, turns, evidence):
            turns[0].speaker_id = "speaker-999"
            raise RuntimeError("private model input")
        self.start("room")
        with patch.object(OnlineDiarizer, "refine", autospec=True, side_effect=broken):
            self.feed()
            result = self.finish()
        self.assertEqual(result.turns[0].speaker_id, "speaker-1")
        self.assertNotIn("private model input", str(result.errors))

    def test_remote_first_numbering_never_collides_with_microphone(self):
        self.manager.embedder_factory = lambda path: lambda pcm: vector(
            2 if pcm.mean() > .15 else (0 if pcm.mean() > 0 else 1))
        self.start("online")
        pcm = np.concatenate((np.full(48000, .1), np.full(48000, -.1))).astype(np.float32)
        self.manager._enqueue(AudioFrame("system", 0, self.manager.started, 16000, pcm))
        self.manager._enqueue(AudioFrame("microphone", 0, self.manager.started, 16000,
                                         np.full(int(16000 * 2), .2, np.float32)))
        result = self.finish()
        self.assertEqual({turn.speaker_id for turn in result.turns}, {"room-1", "call-1", "call-2"})
        self.assertEqual({speaker.name for key, speaker in result.speakers.items() if key != "unknown"},
                         {"В комнате 1", "В созвоне 1", "В созвоне 2"})

    def test_two_local_voices_stay_in_the_room(self):
        self.manager.embedder_factory = lambda path: lambda pcm: vector(0 if pcm.mean() > 0 else 1)
        self.start("online")
        pcm = np.concatenate((np.full(48000, .1), np.full(48000, -.1))).astype(np.float32)
        self.manager._enqueue(AudioFrame("microphone", 0, self.manager.started, 16000, pcm))
        self.manager._enqueue(AudioFrame("system", 1, self.manager.started, 16000,
                                         np.full(32000, -.2, np.float32)))
        result = self.finish()
        mic_turns = [turn for turn in result.turns if turn.source == "microphone"]
        room_ids = {turn.speaker_id for turn in mic_turns if turn.speaker_id.startswith("room-")}
        self.assertEqual(room_ids, {"room-1", "room-2"})
        self.assertTrue(all(not turn.speaker_id.startswith("call-") for turn in mic_turns))

    def test_transcribe_receives_voiced_audio_without_trailing_silence(self):
        lengths = []

        def transcribe(audio):
            lengths.append(len(audio))
            return "Проверка встречи"

        self.engine.transcribe.side_effect = transcribe
        self.start()
        spoken = np.full(16000, .1, np.float32)
        silent = np.zeros(8000, np.float32)
        pcm = np.concatenate((spoken, silent))
        self.manager._enqueue(AudioFrame("microphone", 0, self.manager.started, 16000, pcm))
        result = self.finish()
        self.assertEqual(len(result.turns), 1)
        self.assertEqual(len(lengths), 1)
        self.assertLess(lengths[0], len(pcm))
        self.assertGreater(lengths[0], len(spoken) * .9)
        passed = self.engine.transcribe.call_args[0][0]
        self.assertGreater(np.max(np.abs(passed)), .01)

    def test_short_voiced_mask_skips_transcription(self):
        pcm = np.full(6400, .1, np.float32)
        mask = np.zeros(len(pcm), bool)
        mask[:1600] = True
        piece = SpeechSegment("microphone", 0, .4, pcm, mask)
        self.assertLess(len(voiced_pcm(piece)), MIN_TRANSCRIBE_SAMPLES)

        def split(_self, _segment):
            return [(piece, "speaker-1")]

        with patch.object(OnlineDiarizer, "split", autospec=True, side_effect=split):
            self.start()
            self.feed()
            result = self.finish()
        self.engine.transcribe.assert_not_called()
        self.assertEqual(result.turns, [])

    def test_seventeen_second_audio_yields_two_transcribes(self):
        self.start()
        pcm = np.full(16000 * 17 + 123, .1, np.float32)
        self.manager._enqueue(AudioFrame("microphone", 0, self.manager.started, 16000, pcm))
        result = self.finish()
        self.assertEqual(self.engine.transcribe.call_count, 2)
        self.assertEqual(len(result.turns), 2)

    def test_overlapping_duplicate_online_text_appears_once(self):
        self.manager.embedder_factory = lambda path: lambda pcm: vector()
        sentence = "Один и тот же ответ"
        self.engine.transcribe.return_value = sentence
        self.start("online")
        pcm = np.full(32000, .1, np.float32)
        self.manager._enqueue(AudioFrame("microphone", 0, self.manager.started, 16000, pcm))
        self.manager._enqueue(AudioFrame("system", 0, self.manager.started, 16000,
                                         np.full(32000, -.1, np.float32)))
        result = self.finish()
        self.assertEqual([turn.text for turn in result.turns].count(sentence), 1)
        self.assertEqual(len(result.turns), 1)
        self.assertEqual(result.turns[0].source, "system")

    def test_transcribe_failure_stays_generic(self):
        self.engine.transcribe.side_effect = RuntimeError("secret-pcm-9f3a")
        self.start()
        self.feed()
        result = self.finish()
        joined = " ".join(result.errors)
        self.assertNotIn("secret-pcm-9f3a", joined)
        self.assertIn("Ошибка обработки встречи", joined)

    def test_retired_mode_cannot_open_capture(self):
        with self.assertRaises(ValueError):
            self.manager.start(MeetingOptions("enrollment"))
        self.assertEqual(self.captures, [])

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
    def test_system_capture_does_not_claim_release_on_native_stop_error(self):
        from giga_dictate.system_audio import SystemAudioStream

        capture = SystemAudioStream(Mock(), Mock())
        capture._released.clear()
        capture._started.set()
        capture.receiver = Mock(owner=capture)
        capture.stream = Mock()
        capture.stream.stopCaptureWithCompletionHandler_.side_effect = lambda callback: callback("stop error")
        with self.assertRaisesRegex(RuntimeError, "Не удалось подтвердить"):
            capture.stop()
        self.assertFalse(capture.released)
        self.assertIsNone(capture.receiver)

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
    def test_voice_count_controls_default_to_auto_and_fit_the_view(self):
        from giga_dictate.meeting_ui import meeting_options_view
        view, mode, mic, voices, save = meeting_options_view([(2, "Test microphone")])
        self.assertEqual(voices.titleOfSelectedItem(), "Автоматически")
        self.assertEqual(voices.numberOfItems(), 17)
        voices.selectItemAtIndex_(6)
        self.assertEqual(voices.titleOfSelectedItem(), "6")
        self.assertFalse(save.state())
        for control in view.subviews():
            frame = control.frame()
            self.assertGreaterEqual(frame.origin.x, 0)
            self.assertGreaterEqual(frame.origin.y, 0)
            self.assertLessEqual(frame.origin.x + frame.size.width, view.frame().size.width)
            self.assertLessEqual(frame.origin.y + frame.size.height, view.frame().size.height)

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
        result.speakers = {"speaker-1": Speaker("speaker-1", "Спикер 1"), "speaker-2": Speaker("speaker-2", "Спикер 2")}
        result.turns = [TranscriptTurn(1, 1, 3, "system", "speaker-2", "Привет")]
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
        self.assertIn("Спикер 1: Привет", result.render())
        self.assertNotIn("Вы", [self.ui.speaker_popup.itemTitleAtIndex_(i)
                                  for i in range(self.ui.speaker_popup.numberOfItems())])
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

    def test_menu_has_no_voice_profile_controls(self):
        self.controller.rebuild_menu()
        titles = [str(item.title()) for item in self.controller.menu.itemArray()]
        self.assertFalse(any("мой голос" in title.lower() or "профиль" in title.lower() for title in titles))
        self.assertFalse(hasattr(self.ui, "onEnroll_"))
        self.assertFalse(hasattr(self.ui, "onDeleteProfile_"))

    def test_sleep_only_finishes_active_meeting(self):
        self.ui.session.state = MeetingState.RECORDING
        self.ui.onSleep_(None)
        self.assertTrue(self.ui.session._stop.is_set())
        self.assertIn("сон", self.ui.session._failure)
