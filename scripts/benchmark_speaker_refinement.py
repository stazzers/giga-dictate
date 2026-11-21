#!/usr/bin/env python3
"""Measure current finalization on repeated file-derived acoustic summaries.

Real Silero/CAM++ process one file cycle; only detached summaries are repeated
to the requested duration. This isolates finalization, not full ASR/capture.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from check_speaker_stability import read_pcm
from giga_dictate import __version__
from giga_dictate.diarization import CAMPlusEmbedder, OnlineDiarizer
from giga_dictate.meeting_models import find_models
from giga_dictate.meeting_types import TranscriptTurn
from giga_dictate.vad import SileroProbability, StreamingVAD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=10800)
    parser.add_argument("--audio", type=Path, default=ROOT / "example.wav")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("--seconds must be positive")
    pcm = read_pcm(args.audio)
    if not len(pcm):
        parser.error("Input audio must not be empty")
    models = find_models()
    d = OnlineDiarizer(CAMPlusEmbedder(models / "campplus.onnx"))
    v = StreamingVAD("microphone", SileroProbability(models / "silero_vad.onnx"))
    pieces = []
    for i in range(0, len(pcm), 512):
        for segment in v.feed(i / 16000, pcm[i:i + 512]):
            pieces.extend(d.split(segment))
    for segment in v.flush():
        pieces.extend(d.split(segment))
    turns, evidence = [], {}
    cycle = len(pcm) / 16000
    for repeat in range(math.ceil(args.seconds / cycle)):
        offset = repeat * cycle
        for piece, key in pieces:
            if piece.end + offset > args.seconds:
                continue
            index = len(turns)
            turns.append(TranscriptTurn(index, piece.start + offset, piece.end + offset, piece.source, key, ""))
            item = piece.speaker_evidence
            if item is not None:
                evidence[index] = replace(item, vectors=item.vectors.copy(), weights=item.weights.copy(),
                    intervals=None if item.intervals is None else item.intervals.copy() + offset,
                    quality=None if item.quality is None else item.quality.copy())
    before = time.monotonic()
    changed = d.refine(turns, evidence)
    report = dict(version=__version__, audio_seconds=args.seconds, turns=len(turns),
                  refinement_seconds=round(time.monotonic() - before, 4), changed_turns=changed,
                  speakers=len({t.speaker_id for t in turns if t.speaker_id != "unknown"}),
                  unknown_turns=sum(t.speaker_id == "unknown" for t in turns),
                  representatives=d.refinement_stats.get("representatives", 0),
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  notes="Finalization only; repeated detached file-derived summaries, no capture or ASR.")
    evidence.clear()
    output = json.dumps(report, indent=2) + "\n"
    print(output, end="")
    if args.output:
        args.output.write_text(output)


if __name__ == "__main__":
    main()
