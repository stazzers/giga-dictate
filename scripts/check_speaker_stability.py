#!/usr/bin/env python3
"""Check real local VAD/CAM++ on one-person variations and a two-person control.

Input: two different phrases from one speaker and a phrase from another.
No network, live capture, audio output, or transcript files are used.
This is a small regression check, not an independent DER evaluation.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from giga_dictate.diarization import CAMPlusEmbedder, OnlineDiarizer
from giga_dictate.meeting_models import find_models
from giga_dictate.meeting_types import TranscriptTurn
from giga_dictate.vad import SileroProbability, StreamingVAD


def read_pcm(path):
    with wave.open(str(path)) as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16000):
            raise ValueError("Expected mono PCM16 WAV at 16 kHz")
        return np.frombuffer(handle.readframes(handle.getnframes()), "<i2").astype(np.float32) / 32768


def wrong_speaker_seconds(turns, regions):
    """Assigned-to-wrong-person time on constructed regions; not a DER score.

    Ignore unknown and a 250 ms collar. Map speaker numbers optimally.
    """
    found = sorted({turn.speaker_id for turn in turns if turn.speaker_id != "unknown"})
    people = sorted({person for _, _, person in regions})
    weights = np.zeros((len(people), len(found)))
    for start, end, person in regions:
        for turn in turns:
            if turn.speaker_id in found:
                overlap = max(0, min(end - .25, turn.end) - max(start + .25, turn.start))
                weights[people.index(person), found.index(turn.speaker_id)] += overlap

    @lru_cache(None)
    def matching(i, used):
        if i == len(people):
            return 0.0
        return max([matching(i + 1, used), *[
            weights[i, j] + matching(i + 1, used | (1 << j))
            for j in range(len(found)) if not used & (1 << j)]])

    return round(float(weights.sum() - matching(0, 0)), 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same-person", nargs=2, required=True, type=Path)
    parser.add_argument("--other-person", required=True, type=Path)
    parser.add_argument("--russian", type=Path, default=ROOT / "example.wav")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    models = find_models()
    embedder = CAMPlusEmbedder(models / "campplus.onnx")
    a = read_pcm(args.russian)
    x, y = map(read_pcm, args.same_person)
    b = read_pcm(args.other_person)
    silence = np.zeros(8000, np.float32)
    rng = np.random.default_rng(42)
    cases = [
        ("russian_clean", a, 1),
        ("russian_quiet", a * .12, 1),
        ("russian_noisy", a + rng.normal(0, .003, len(a)).astype(np.float32), 1),
        ("russian_echo", a + np.pad(a[:-1200] * .15, (1200, 0)), 1),
        ("same_person_different_phrases", np.concatenate((x, silence, y)), 1),
        ("same_person_longer", np.tile(np.concatenate((a, silence)), 6), 1),
        ("two_people", np.concatenate((x, silence, b)), 2),
        ("two_people_without_pause", np.concatenate((x, b)), 2),
        ("two_people_alternating", np.concatenate((x, silence, b, silence, y, silence, b)), 2),
    ]
    references = {}

    def conversation(name, parts, expected):
        position, regions, audio = 0.0, [], []
        for pcm, person in parts:
            audio.extend((pcm, silence))
            regions.append((position, position + len(pcm) / 16000, person))
            position += (len(pcm) + len(silence)) / 16000
        cases.append((name, np.concatenate(audio), expected))
        references[name] = regions

    conversation("russian_short_then_long", [(a[8000:20800], 1), (a[32000:44800], 1),
                                             (a[64000:76800], 1), (a, 1)], 1)
    conversation("two_people_short_then_long", [(x[8000:20800], 1), (b[8000:20800], 2),
                                                (x[32000:44800], 1), (x, 1), (b, 2)], 2)
    conversation("unconfirmed_other_voice", [(b[8000:20800], 2), (x, 1)], 1)
    conversation("two_people_control", [(x, 1), (b, 2), (y, 1), (b, 2)], 2)
    rows = []
    for name, pcm, expected in cases:
        vad = StreamingVAD("microphone", SileroProbability(models / "silero_vad.onnx"))
        diarizer = OnlineDiarizer(embedder)
        pieces = []
        # Exercise real streaming block boundaries, not just a whole-file call.
        for start in range(0, len(pcm), 512):
            for segment in vad.feed(start / 16000, pcm[start:start + 512]):
                pieces.extend(diarizer.split(segment))
        for segment in vad.flush():
            pieces.extend(diarizer.split(segment))
        turns = [TranscriptTurn(i, piece.start, piece.end, piece.source, key, "")
                 for i, (piece, key) in enumerate(pieces)]
        before_turns = [TranscriptTurn(turn.id, turn.start, turn.end, turn.source,
                                     diarizer.resolve_speaker(turn.speaker_id), "") for turn in turns]
        before = sum(turn.end - turn.start for turn in turns
                     if diarizer.resolve_speaker(turn.speaker_id) == "unknown")
        evidence = {i: piece.speaker_evidence for i, (piece, _) in enumerate(pieces)
                    if piece.speaker_evidence is not None}
        started = time.monotonic()
        refined = diarizer.refine(turns, evidence)
        elapsed = time.monotonic() - started
        labels = [(turn.end - turn.start, turn.speaker_id) for turn in turns]
        found = len({key for _, key in labels if key != "unknown"})
        rows.append(dict(case=name, expected=expected, speakers=found,
                         unknown_before=round(before, 2), refined_turns=refined,
                         refinement_ms=round(elapsed * 1000, 2),
                         unknown_seconds=round(sum(length for length, key in labels if key == "unknown"), 2),
                         speech_seconds=round(sum(length for length, _ in labels), 2)))
        if name in references:
            rows[-1]["wrong_before"] = wrong_speaker_seconds(before_turns, references[name])
            rows[-1]["wrong_after"] = wrong_speaker_seconds(turns, references[name])
    output = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    print(output, end="")
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    return int(any(row["expected"] != row["speakers"] or
                   row.get("wrong_after", 0) > row.get("wrong_before", 0) + .05 for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
