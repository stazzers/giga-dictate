"""Distinguish sustained new people from isolated bad embedding windows."""
import numpy as np
import unittest
from unittest.mock import Mock

from giga_dictate.diarization import OnlineDiarizer
from giga_dictate.meeting_types import SpeakerEvidence, SpeechSegment, TranscriptTurn
from giga_dictate.vad import StreamingVAD
from tests.test_meeting import vector, Probability


class SpeakerStability(unittest.TestCase):
    def evidence(self, vector, seconds=2.0):
        return SpeakerEvidence(np.asarray([vector], np.float32), np.array([seconds], np.float32), seconds)

    def confirm_two(self, diarizer):
        first = diarizer.identify(vector())
        for _ in range(3):
            second = diarizer.identify(vector(1))
        return first, second

    def test_final_pass_uses_later_voice_examples_to_recover_earlier_phrase(self):
        diarizer = OnlineDiarizer(lambda pcm: vector())
        first = diarizer.identify(vector())
        early = vector() * .4 + vector(2) * np.sqrt(1 - .4 ** 2)
        self.assertEqual(diarizer.identify(early), "unknown")
        later = vector() * .6 + vector(2) * .8
        self.assertEqual(diarizer.identify(later), first)
        turn = TranscriptTurn(1, 1, 3, "microphone", "unknown", "Ранняя реплика", "uncertain")
        self.assertEqual(diarizer.refine([turn], {1: self.evidence(early)}), 1)
        self.assertEqual(turn.speaker_id, first)

    def test_final_pass_requires_a_clear_winner_between_people(self):
        diarizer = OnlineDiarizer(lambda pcm: vector())
        first, second = self.confirm_two(diarizer)
        unclear = vector() * .6 + vector(1) * .6 + vector(2) * np.sqrt(.28)
        turns = [TranscriptTurn(1, 0, 2, "microphone", first, ""),
                 TranscriptTurn(2, 2, 4, "microphone", "unknown", ""),
                 TranscriptTurn(3, 4, 6, "microphone", second, "")]
        self.assertEqual(diarizer.refine(turns, {2: self.evidence(unclear)}), 0)
        self.assertEqual([t.speaker_id for t in turns], [first, "unknown", second])

    def test_final_pass_recovers_unique_moderate_match_without_merging_people(self):
        diarizer = OnlineDiarizer(lambda pcm: vector())
        first, second = self.confirm_two(diarizer)
        moderate = vector() * .45 + vector(2) * np.sqrt(1 - .45 ** 2)
        turn = TranscriptTurn(1, 0, 2, "microphone", "unknown", "")
        self.assertEqual(diarizer.refine([turn], {1: self.evidence(moderate)}), 1)
        self.assertEqual(turn.speaker_id, first)
        self.assertEqual(set(diarizer.centers), {first, second})

    def test_unknown_reply_can_be_assigned_to_a_person_confirmed_later(self):
        diarizer = OnlineDiarizer(lambda pcm: vector())
        self.confirm_two(diarizer)
        turn = TranscriptTurn(1, 0, .8, "system", "unknown", "")
        diarizer.refine([turn], {1: self.evidence(vector(1), .8)})
        self.assertEqual(turn.speaker_id, "speaker-2")

    def test_short_consistent_replies_pool_acoustic_evidence(self):
        diarizer = OnlineDiarizer(lambda pcm: vector())
        first, _ = self.confirm_two(diarizer)
        moderate = vector() * .5 + vector(2) * np.sqrt(.75)
        one = TranscriptTurn(1, 0, .7, "microphone", "unknown", "")
        diarizer.refine([one], {1: self.evidence(moderate, .7)})
        self.assertEqual(one.speaker_id, "unknown")
        two = TranscriptTurn(2, 1, 1.7, "microphone", "unknown", "")
        diarizer.refine([one, two], {1: self.evidence(moderate, .7), 2: self.evidence(moderate, .7)})
        self.assertEqual([one.speaker_id, two.speaker_id], [first, first])

    def test_short_reply_matches_only_with_a_clear_lead(self):
        diarizer = OnlineDiarizer(lambda pcm: vector())
        first, _ = self.confirm_two(diarizer)
        match = vector() * .58 + vector(2) * np.sqrt(1 - .58 ** 2)
        ambiguous = vector() * .61 + vector(1) * .45 + vector(3) * np.sqrt(1 - .61 ** 2 - .45 ** 2)
        turns = [TranscriptTurn(1, 0, .8, "microphone", "unknown", ""),
                 TranscriptTurn(2, 4, 4.8, "microphone", "unknown", "")]
        diarizer.refine(turns, {1: self.evidence(match, .8), 2: self.evidence(ambiguous, .8)})
        self.assertEqual([turn.speaker_id for turn in turns], [first, "unknown"])

    def test_no_assignment_from_neighbours_without_acoustic_evidence(self):
        diarizer = OnlineDiarizer(lambda pcm: vector())
        first, second = self.confirm_two(diarizer)
        turn = TranscriptTurn(1, 0, .3, "microphone", "unknown", "")
        unrelated = TranscriptTurn(2, 1, 3, "microphone", "unknown", "")
        diarizer.refine([turn, unrelated], {2: self.evidence(vector(2))})
        self.assertEqual([turn.speaker_id, unrelated.speaker_id], ["unknown", "unknown"])
        self.assertEqual(set(diarizer.centers), {first, second})

    def test_manual_assignments_and_online_microphone_are_preserved(self):
        call = OnlineDiarizer(lambda pcm: vector(), id_prefix="call", name_prefix="В созвоне")
        remote, other = self.confirm_two(call)
        local = TranscriptTurn(1, 0, 2, "microphone", "room-1", "")
        manual = TranscriptTurn(2, 0, 2, "system", "unknown", "", manual_speaker_id=other)
        call.refine([manual], {2: self.evidence(vector())})
        self.assertEqual(local.speaker_id, "room-1")
        self.assertEqual(manual.effective_speaker_id, other)
        self.assertNotEqual(manual.effective_speaker_id, remote)

    def test_acoustic_summary_is_small_and_does_not_keep_pcm(self):
        pcm = np.full(15 * 16000, .1, np.float32)
        diarizer = OnlineDiarizer(lambda speech: vector())
        pieces = diarizer.split(SpeechSegment("microphone", 0, 15, pcm))
        for piece, _ in pieces:
            item = piece.speaker_evidence
            self.assertLessEqual(len(item.vectors), 8)
            self.assertEqual(item.vectors.shape[1], 192)
            self.assertFalse(np.shares_memory(item.vectors, pcm))
            self.assertNotIn("array", repr(item))

    def test_subsecond_phrase_keeps_evidence_for_final_pass(self):
        diarizer = OnlineDiarizer(lambda speech: vector())
        pieces = diarizer.split(SpeechSegment("microphone", 0, .8, np.full(12800, .1, np.float32)))
        piece, key = pieces[0]
        self.assertEqual(key, "unknown")
        self.assertIsNotNone(piece.speaker_evidence)
        self.assertAlmostEqual(piece.speaker_evidence.speech_seconds, .8)

    def test_weak_final_assignments_do_not_train_other_weak_assignments(self):
        diarizer = OnlineDiarizer(lambda pcm: vector())
        first, _ = self.confirm_two(diarizer)
        moderate = vector() * .45 + vector(2) * np.sqrt(1 - .45 ** 2)
        unrelated = vector() * .1 + vector(2) * np.sqrt(.99)
        turns = [TranscriptTurn(1, 0, 2, "microphone", "unknown", ""),
                 TranscriptTurn(2, 3, 5, "microphone", "unknown", "")]
        diarizer.refine(turns, {1: self.evidence(moderate), 2: self.evidence(unrelated)})
        self.assertEqual([turn.speaker_id for turn in turns], [first, "unknown"])

    def test_single_bad_window_does_not_become_a_person(self):
        diarizer = OnlineDiarizer(lambda pcm: vector())
        first = diarizer.identify(vector())
        candidate = diarizer.identify(vector(1))
        self.assertEqual(diarizer.resolve_speaker(candidate), "unknown")
        self.assertEqual(diarizer.identify(vector()), first)
        self.assertEqual(len(diarizer.centers), 1)
        self.assertEqual(len(diarizer.speakers), 2)  # one person plus unknown

    def test_second_person_requires_consistent_evidence_and_resolves_earlier_turns(self):
        diarizer = OnlineDiarizer(lambda pcm: vector())
        first = diarizer.identify(vector())
        candidate = diarizer.identify(vector(1))
        self.assertNotIn(candidate, diarizer.speakers)
        diarizer.identify(vector(1))
        second = diarizer.identify(vector(1))
        self.assertNotEqual(first, second)
        self.assertEqual(diarizer.resolve_speaker(candidate), second)
        self.assertEqual(len(diarizer.centers), 2)

    def test_inconsistent_outliers_do_not_create_multiple_people(self):
        diarizer = OnlineDiarizer(lambda pcm: vector())
        first = diarizer.identify(vector())
        for i in range(1, 10):
            diarizer.identify(vector(i))
            self.assertEqual(diarizer.identify(vector()), first)
        self.assertEqual(len(diarizer.centers), 1)

    def test_voice_gallery_remembers_phonetic_variants(self):
        diarizer = OnlineDiarizer(lambda pcm: vector())
        first = diarizer.identify(vector())
        alternate = vector() * .6 + vector(1) * .8
        self.assertEqual(diarizer.identify(alternate), first)
        for _ in range(30):
            self.assertEqual(diarizer.identify(vector()), first)
        self.assertEqual(diarizer.identify(alternate), first)
        self.assertLessEqual(len(diarizer.references[first]), 8)

    def test_silence_padding_never_enters_embedding(self):
        pcm = np.concatenate((np.full(32000, .1), np.zeros(16000))).astype(np.float32)
        mask = pcm != 0
        embed = Mock(return_value=vector())
        diarizer = OnlineDiarizer(embed)
        diarizer.split(SpeechSegment("microphone", 0, 3, pcm, mask))
        self.assertGreater(embed.call_count, 0)
        for call in embed.call_args_list:
            self.assertTrue(np.all(call.args[0] == .1))

    def test_vad_tracks_mask_through_final_partial_frame(self):
        pcm = np.concatenate((np.zeros(4000), np.full(16000, .1), np.zeros(9000))).astype(np.float32)
        vad = StreamingVAD("microphone", Probability())
        segments = vad.feed(0, pcm) + vad.flush()
        self.assertTrue(segments)
        for segment in segments:
            self.assertEqual(len(segment.pcm), len(segment.speech_mask))
            self.assertLess(np.count_nonzero(segment.speech_mask), len(segment.pcm))
