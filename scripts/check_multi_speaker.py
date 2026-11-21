#!/usr/bin/env python3
"""Offline CAM++/Silero checks from explicitly supplied per-person PCM16 WAVs.

Use different input recordings for development and validation. Constructed
conversations are regression fixtures, never an independent meeting corpus.
Only aggregate metrics are printed/written; no audio or transcript is exported.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from check_speaker_stability import read_pcm, wrong_speaker_seconds
from evaluate_meeting import evaluate
from giga_dictate.diarization import CAMPlusEmbedder, OnlineDiarizer, SpeakerCountError
from giga_dictate.meeting_models import find_models
from giga_dictate.meeting_types import TranscriptTurn
from giga_dictate.vad import SileroProbability, StreamingVAD


def run_case(pcm, reference, embedder, models, factory, count=None):
    vad = StreamingVAD("microphone", SileroProbability(models / "silero_vad.onnx"))
    diarizer = factory(embedder) if count is None else factory(embedder, speaker_count=count)
    pieces = []
    for start in range(0, len(pcm), 512):
        for segment in vad.feed(start / 16000, pcm[start:start + 512]):
            pieces.extend(diarizer.split(segment))
    for segment in vad.flush():
        pieces.extend(diarizer.split(segment))
    turns = [TranscriptTurn(i, p.start, p.end, p.source, key, "") for i, (p, key) in enumerate(pieces)]
    evidence = {i: p.speaker_evidence for i, (p, _) in enumerate(pieces) if p.speaker_evidence is not None}
    started = time.monotonic()
    count_warning = False
    try:
        diarizer.refine(turns, evidence)
    except SpeakerCountError:
        count_warning = True
        for turn in turns:
            turn.speaker_id = diarizer.resolve_speaker(turn.speaker_id)
    elapsed = time.monotonic() - started
    hypothesis = [dict(source=t.source, start=t.start, end=t.end, speaker_id=t.speaker_id, text="") for t in turns]
    regions = [(r["start"], r["end"], r["speaker_id"]) for r in reference]
    metrics = evaluate(reference, hypothesis)
    return dict(speakers=len({t.speaker_id for t in turns if t.speaker_id != "unknown"}),
                count_warning=count_warning,
                wrong_seconds=wrong_speaker_seconds(turns, regions),
                unknown_seconds=round(sum(max(0., min(t.end, r["end"]) - max(t.start, r["start"]))
                                         for t in turns if t.speaker_id == "unknown" for r in reference), 3),
                non_overlap_der=metrics["non_overlap_der"],
                refinement_ms=round(elapsed * 1000, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", type=Path, action="append", required=True,
                        help="One recording per different voice; repeat 3–15 times")
    parser.add_argument("--baseline-module", type=Path)
    parser.add_argument("--partition", choices=("development", "validation"), default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 3 <= len(args.speaker) <= 15:
        parser.error("Supply 3–15 different speakers")
    audio = [read_pcm(path) for path in args.speaker]
    if any(len(pcm) < 8 * 16000 for pcm in audio):
        parser.error("Every recording must have at least eight seconds")
    models = find_models()
    embedder = CAMPlusEmbedder(models / "campplus.onnx")
    baseline = None
    if args.baseline_module:
        spec = importlib.util.spec_from_file_location("giga_dictate._baseline_diarization", args.baseline_module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        baseline = module.OnlineDiarizer
    reports = []
    for count in sorted({3, min(5, len(audio)), min(6, len(audio))}):
        chunks, reference, position = [], [], 0.

        def append(pcm, person):
            nonlocal position
            reference.append(dict(source="microphone", start=position, end=position + len(pcm) / 16000,
                                  speaker_id=str(person), text=""))
            chunks.extend((pcm, np.zeros(16000, np.float32)))
            position += len(pcm) / 16000 + 1.

        # Only the first two voices get long introductions. Other participants
        # must be discovered from separated short replies, including at the end.
        for person in range(2):
            append(audio[person][:64000], person)
        for repeat in range(6):
            for person in range(count):
                start = int((1. + repeat * .9) * 16000)
                append(audio[person][start:start + 14400], person)
        pcm = np.concatenate(chunks)
        row = dict(expected=count, audio_seconds=round(len(pcm) / 16000, 2),
                   automatic=run_case(pcm, reference, embedder, models, OnlineDiarizer),
                   exact=run_case(pcm, reference, embedder, models, OnlineDiarizer, count))
        if baseline:
            row["baseline"] = run_case(pcm, reference, embedder, models, baseline)
        reports.append(row)
    report = dict(partition=args.partition, cases=reports,
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  notes="Constructed file regression; not independent human-meeting quality or overlap-aware DER.")
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(output, end="")
    if args.output:
        args.output.write_text(output)
    return int(any(row[mode]["speakers"] != row["expected"] or
                   ("baseline" in row and row[mode]["wrong_seconds"] > row["baseline"]["wrong_seconds"] + .05)
                   for row in reports for mode in ("automatic", "exact")))


if __name__ == "__main__":
    raise SystemExit(main())
