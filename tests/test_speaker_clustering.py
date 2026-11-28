"""Meeting-wide clustering regressions; no devices, network or private audio."""
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from giga_dictate.diarization import OnlineDiarizer, SpeakerCountError, suppress_remote_echo
from giga_dictate.meeting_types import MeetingOptions, SpeakerEvidence, SpeechSegment, TranscriptTurn
from giga_dictate.speaker_clustering import average_linkage, stable_mapping
from tests.test_meeting import vector


def summary(value, start, seconds=.8, quality=1.):
    return SpeakerEvidence(value[None], np.array([seconds], np.float32), seconds,
                           np.array([[start, start + seconds]]), np.array([quality]))


def conversation(people, seconds=.8, labels=None, source="microphone"):
    turns, evidence = [], {}
    for i, person in enumerate(people):
        start = i * (seconds + .5)
        label = labels[i] if labels is not None else "unknown"
        turns.append(TranscriptTurn(i, start, start + seconds, source, label, "Тест"))
        evidence[i] = summary(vector(person), start, seconds)
    return turns, evidence


class GlobalSpeakers(TestCase):
    def test_candidates_survive_other_people_and_confirm_independently(self):
        d = OnlineDiarizer(None)
        first = d.identify(vector())
        pending = {}
        for repeat in range(3):
            for person in (1, 2):
                label = d.identify(vector(person))
                pending.setdefault(person, label)
                self.assertEqual(d.identify(vector()), first)
        self.assertEqual(len(d.centers), 3)
        self.assertNotEqual(d.resolve_speaker(pending[1]), d.resolve_speaker(pending[2]))
        self.assertNotIn("unknown", [d.resolve_speaker(k) for k in pending.values()])

    def test_overlapping_windows_do_not_count_as_independent_speech(self):
        d = OnlineDiarizer(None)
        d.identify(vector())
        for i in range(20):
            label = d.identify(vector(1), start=i * .01, end=2 + i * .01, seconds=2.)
        self.assertEqual(d.resolve_speaker(label), "unknown")
        self.assertEqual(len(d.centers), 1)
        d.identify(vector(1), start=4, end=5, seconds=1.)
        self.assertNotEqual(d.resolve_speaker(label), "unknown")

    def test_third_voice_with_only_separated_short_replies(self):
        d = OnlineDiarizer(None)
        first = d.identify(vector())
        for _ in range(3):
            d.identify(vector(1))
        turns, evidence = conversation([0, 2, 1, 2, 0, 2, 1, 2])
        d.refine(turns, evidence)
        self.assertEqual({t.speaker_id for t in turns}, {"speaker-1", "speaker-2", "speaker-3"})
        self.assertTrue(all(t.speaker_id == "speaker-3" for t in turns if t.id in (1, 3, 5, 7)))
        self.assertEqual(turns[0].speaker_id, first)

    def test_auto_and_exact_three_five_six_voices(self):
        for count in (3, 5, 6):
            for requested in (None, count):
                with self.subTest(count=count, requested=requested):
                    d = OnlineDiarizer(None, speaker_count=requested)
                    people = list(range(count)) * 4
                    turns, evidence = conversation(people)
                    d.refine(turns, evidence)
                    labels = [t.speaker_id for t in turns[:count]]
                    self.assertEqual(len(set(labels)), count)
                    self.assertNotIn("unknown", labels)
                    self.assertEqual([t.speaker_id for t in turns], labels * 4)

    def test_late_new_voice_can_be_confirmed_after_capture_stops(self):
        d = OnlineDiarizer(None)
        d.identify(vector())
        turns, evidence = conversation([0, 0, 0, 2], seconds=3.)
        d.refine(turns, evidence)
        self.assertNotEqual(turns[-1].speaker_id, turns[0].speaker_id)
        self.assertNotEqual(turns[-1].speaker_id, "unknown")

    def test_wrong_automatic_assignment_is_revisited(self):
        d = OnlineDiarizer(None)
        d.identify(vector())
        turns, evidence = conversation([0, 1, 0, 1, 0, 1, 0, 1], labels=["speaker-1"] * 8)
        d.refine(turns, evidence)
        self.assertEqual(len({t.speaker_id for t in turns}), 2)
        self.assertEqual(turns[0].speaker_id, "speaker-1")
        self.assertNotEqual(turns[1].speaker_id, "speaker-1")

    def test_duplicate_confirmed_people_merge_and_keep_dominant_id(self):
        d = OnlineDiarizer(None)
        d.identify(vector())
        for _ in range(3):
            d.identify(vector(1))
        d.references["speaker-2"] = [vector()]
        d.centers["speaker-2"] = vector()
        d.aliases["pending-old"] = "speaker-2"
        turns, evidence = conversation([0, 0, 0, 0], labels=["speaker-1"] * 3 + ["speaker-2"])
        d.refine(turns, evidence)
        self.assertEqual({t.speaker_id for t in turns}, {"speaker-1"})
        self.assertEqual(d.resolve_speaker("pending-old"), "speaker-1")

    def test_single_outlier_and_low_quality_replies_do_not_create_people(self):
        d = OnlineDiarizer(None)
        d.identify(vector())
        turns, evidence = conversation([0, 1, 2, 2, 2, 2])
        for i in range(2, 6):
            evidence[i].quality[:] = .4
        d.refine(turns, evidence)
        self.assertEqual(turns[0].speaker_id, "speaker-1")
        self.assertTrue(all(t.speaker_id == "unknown" for t in turns[1:]))
        self.assertEqual(len(d.centers), 1)

    def test_duplicate_windows_and_turns_do_not_create_new_people(self):
        d = OnlineDiarizer(None)
        d.identify(vector())
        turns, evidence = conversation([1, 1, 1, 1])
        for turn in turns:
            turn.start, turn.end = 0., .8
            evidence[turn.id] = summary(vector(1), 0.)
        d.refine(turns, evidence)
        self.assertTrue(all(t.speaker_id == "unknown" for t in turns))

    def test_manual_id_and_renamed_speaker_survive_clustering(self):
        d = OnlineDiarizer(None)
        first = d.identify(vector())
        d.speakers[first].name = "Алексей"
        turns, evidence = conversation([0, 1, 0, 1, 0, 1, 1])
        turns[1].manual_speaker_id = first
        d.refine(turns, evidence)
        self.assertEqual(turns[1].effective_speaker_id, first)
        self.assertEqual(d.speakers[first].name, "Алексей")

    def test_online_galleries_do_not_mix_across_diarizers(self):
        call = OnlineDiarizer(None, id_prefix="call", name_prefix="В созвоне", speaker_count=2)
        turns, evidence = conversation([0, 1] * 4, source="system")
        local = TranscriptTurn(99, 0, 3, "microphone", "room-1", "Тест")
        call.refine(turns, evidence)
        self.assertEqual(local.speaker_id, "room-1")
        self.assertEqual({t.speaker_id for t in turns}, {"call-1", "call-2"})

    def test_exact_count_can_split_similar_voices(self):
        d = OnlineDiarizer(None, speaker_count=2)
        turns, evidence = conversation([0, 1] * 4, seconds=2.)
        similar = vector() * .8 + vector(1) * .6
        for i in (1, 3, 5, 7):
            evidence[i] = summary(similar, turns[i].start, 2.)
        d.refine(turns, evidence)
        self.assertEqual(len({t.speaker_id for t in turns}), 2)
        self.assertNotEqual(turns[0].speaker_id, turns[1].speaker_id)

    def test_unachievable_count_preserves_turns_galleries_and_names(self):
        for count in (2, 6):
            d = OnlineDiarizer(None, speaker_count=count)
            first = d.identify(vector())
            turns, evidence = conversation([0] * 4, labels=[first] * 4)
            old_speakers = dict(d.speakers)
            old_center = d.centers[first].copy()
            with self.assertRaises(SpeakerCountError):
                d.refine(turns, evidence)
            self.assertTrue(all(t.speaker_id == first for t in turns))
            self.assertEqual(d.speakers, old_speakers)
            np.testing.assert_array_equal(d.centers[first], old_center)

    def test_fixed_count_does_not_spend_a_cluster_on_an_isolated_outlier(self):
        d = OnlineDiarizer(None, speaker_count=2)
        turns, evidence = conversation([0, 1] * 4 + [2])
        related = vector() * .35 + vector(1) * np.sqrt(1 - .35 ** 2)
        for i in (1, 3, 5, 7):
            evidence[i] = summary(related, turns[i].start)
        d.refine(turns, evidence)
        self.assertEqual(len({t.speaker_id for t in turns[:-1]}), 2)
        self.assertNotEqual(turns[0].speaker_id, turns[1].speaker_id)
        self.assertEqual(turns[-1].speaker_id, "unknown")

    def test_one_requested_cluster_can_contain_opposite_vectors(self):
        d = OnlineDiarizer(None, speaker_count=1)
        turns, evidence = conversation([0, 0], seconds=3.)
        evidence[1] = summary(-vector(), turns[1].start, 3.)
        d.refine(turns, evidence)
        self.assertEqual(len({t.speaker_id for t in turns}), 1)

    def test_error_after_clustering_cannot_partially_commit(self):
        d = OnlineDiarizer(None)
        turns, evidence = conversation([0, 1, 2] * 4)
        with patch.object(OnlineDiarizer, "_refine_known", side_effect=RuntimeError("failure")):
            with self.assertRaises(RuntimeError):
                d.refine(turns, evidence)
        self.assertEqual(d.centers, {})
        self.assertEqual(set(d.speakers), {"unknown"})
        self.assertTrue(all(t.speaker_id == "unknown" for t in turns))

    def test_invalid_intervals_fail_atomically(self):
        d = OnlineDiarizer(None)
        turns, evidence = conversation([0])
        evidence[0].intervals[0, 1] = 10
        with self.assertRaises(ValueError):
            d.refine(turns, evidence)
        self.assertEqual(d.centers, {})

    def test_summary_intervals_are_inside_pieces(self):
        d = OnlineDiarizer(lambda pcm: vector(0 if pcm.mean() > 0 else 1))
        pcm = np.r_[np.full(48000, .1), np.full(48000, -.1)].astype(np.float32)
        pieces = d.split(SpeechSegment("microphone", 10, 16, pcm))
        self.assertEqual(sum(len(p.pcm) for p, _ in pieces), len(pcm))
        for p, _ in pieces:
            item = p.speaker_evidence
            self.assertTrue(np.all(item.intervals[:, 0] >= p.start))
            self.assertTrue(np.all(item.intervals[:, 1] <= p.end))

    def test_subsecond_reply_waits_for_final_independent_matching(self):
        d = OnlineDiarizer(lambda pcm: vector())
        first = d.identify(vector())
        pieces = d.split(SpeechSegment("microphone", 5, 5.8, np.full(12800, .1, np.float32)))
        piece, provisional = pieces[0]
        self.assertEqual(provisional, "unknown")
        self.assertEqual(d.counts[first], 1)
        turn = TranscriptTurn(1, piece.start, piece.end, piece.source, provisional, "Тест")
        d.refine([turn], {1: piece.speaker_evidence})
        self.assertEqual(turn.speaker_id, first)


class ClusteringContracts(TestCase):
    def test_options_validate_counts_and_keep_positional_compatibility(self):
        self.assertIsNone(MeetingOptions("room", None, None).speaker_count)
        for mode, limit in (("room", 15), ("online", 16)):
            for count in (None, 1, limit):
                MeetingOptions(mode, speaker_count=count).validate()
            for count in (True, 0, -1, 1.5, "3", limit + 1):
                with self.assertRaises(ValueError):
                    MeetingOptions(mode, speaker_count=count).validate()

    def test_average_linkage_auto_and_exact_cut(self):
        vectors = np.stack([vector(), vector(), vector(1), vector(1)])
        self.assertEqual(average_linkage(vectors, [1] * 4), [[0, 1], [2, 3]])
        self.assertEqual(len(average_linkage(vectors, [1] * 4, count=3)), 3)

    def test_id_matching_is_globally_optimal_not_greedy(self):
        self.assertEqual(stable_mapping([[9, 8], [8, 0]], ["a", "b"]), {1: "a", 0: "b"})
        self.assertEqual(stable_mapping([[0, 0]], ["a", "b"]), {})

    def test_representatives_are_bounded(self):
        d = OnlineDiarizer(None)
        rng = np.random.default_rng(17)
        turns, evidence = conversation([0] * 1100)
        for i, turn in enumerate(turns):
            v = rng.normal(size=192).astype(np.float32)
            evidence[i] = summary(v / np.linalg.norm(v), turn.start)
        from giga_dictate.diarization import average_linkage as real_linkage
        def check(vectors, *args, **kwargs):
            self.assertLessEqual(len(vectors), 1024)
            return real_linkage(vectors, *args, **kwargs)
        with patch("giga_dictate.diarization.average_linkage", side_effect=check) as cluster:
            d.refine(turns, evidence)
        cluster.assert_called_once()


class RemoteEchoSuppression(TestCase):
    def setUp(self):
        self.call = OnlineDiarizer(None, id_prefix="call", name_prefix="В созвоне")
        self.call.identify(vector())

    def test_duplicate_text_removes_microphone_turn(self):
        turns = [TranscriptTurn(1, 0, 2, "system", "call-1", "Привет как дела"),
                 TranscriptTurn(2, .1, 2.1, "microphone", "room-2", "Привет как дела")]
        evidence = {1: summary(vector(), 0, 2), 2: summary(vector(), .1, 2)}
        suppress_remote_echo(turns, evidence, self.call)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].speaker_id, "call-1")

    def test_different_text_reassigns_to_call_speaker(self):
        turns = [TranscriptTurn(1, 0, 2, "system", "call-1", "Привет как дела"),
                 TranscriptTurn(2, .1, 2.1, "microphone", "room-2", "Другая фраза")]
        evidence = {1: summary(vector(), 0, 2), 2: summary(vector(), .1, 2)}
        suppress_remote_echo(turns, evidence, self.call)
        self.assertEqual(len(turns), 2)
        mic = next(turn for turn in turns if turn.source == "microphone")
        self.assertEqual(mic.speaker_id, "call-1")
        self.assertFalse(mic.speaker_id.startswith("room-"))

    def test_late_microphone_without_overlap_stays_room(self):
        turns = [TranscriptTurn(1, 0, 2, "system", "call-1", "Привет как дела"),
                 TranscriptTurn(2, 2.5, 4.5, "microphone", "room-1", "Привет как дела")]
        evidence = {1: summary(vector(), 0, 2), 2: summary(vector(), 2.5, 2)}
        suppress_remote_echo(turns, evidence, self.call)
        mic = next(turn for turn in turns if turn.source == "microphone")
        self.assertEqual(mic.speaker_id, "room-1")

    def test_mismatched_embedding_stays_room(self):
        turns = [TranscriptTurn(1, 0, 2, "system", "call-1", "Привет как дела"),
                 TranscriptTurn(2, .1, 2.1, "microphone", "room-1", "Привет как дела")]
        evidence = {1: summary(vector(), 0, 2), 2: summary(vector(1), .1, 2)}
        suppress_remote_echo(turns, evidence, self.call)
        mic = next(turn for turn in turns if turn.source == "microphone")
        self.assertEqual(mic.speaker_id, "room-1")

    def test_manual_microphone_assignment_is_preserved(self):
        turns = [TranscriptTurn(1, 0, 2, "system", "call-1", "Привет как дела"),
                 TranscriptTurn(2, .1, 2.1, "microphone", "room-2", "Привет как дела",
                               manual_speaker_id="room-9")]
        evidence = {1: summary(vector(), 0, 2), 2: summary(vector(), .1, 2)}
        suppress_remote_echo(turns, evidence, self.call)
        mic = next(turn for turn in turns if turn.source == "microphone")
        self.assertEqual(mic.speaker_id, "room-2")
        self.assertEqual(mic.manual_speaker_id, "room-9")
