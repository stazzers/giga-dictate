#!/usr/bin/env python3
"""Compute WER and non-overlap DER from explicitly supplied annotated JSON.

Inputs are lists of {source, start, end, speaker_id, text}. No text is printed.
DER uses a 10 ms grid, reference collars and optimal speaker mapping per source.
Overlapping speech is excluded; this is not an overlap-aware diarization score.
"""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import re

import numpy as np


def word_errors(reference, hypothesis):
    words = lambda text: re.findall(r"\w+", text.lower().replace("ё", "е"))
    ref, hyp = words(reference), words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        current = [i]
        for j, other in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (word != other)))
        previous = current
    return previous[-1], len(ref)


def evaluate(reference, hypothesis, collar=.25):
    for row in [*reference, *hypothesis]:
        if not (math.isfinite(row["start"]) and math.isfinite(row["end"]) and
                0 <= row["start"] < row["end"] <= 86400):
            raise ValueError("Invalid annotation timestamps")
    errors = words = missed = false_alarm = confused = speech = scored = 0
    for source in {row["source"] for row in [*reference, *hypothesis]}:
        ref = sorted([row for row in reference if row["source"] == source], key=lambda row: row["start"])
        hyp = sorted([row for row in hypothesis if row["source"] == source], key=lambda row: row["start"])
        e, count = word_errors(" ".join(row["text"] for row in ref), " ".join(row["text"] for row in hyp))
        errors += e
        words += count
        duration = max((row["end"] for row in [*ref, *hyp]), default=0)
        size = math.ceil(duration * 100)
        ref_names = sorted({row["speaker_id"] for row in ref})
        hyp_names = sorted({row["speaker_id"] for row in hyp if row["speaker_id"] != "unknown"})
        if max(len(ref_names), len(hyp_names)) > 15:
            raise ValueError("At most 15 speakers per source are supported")

        def raster(rows, names):
            labels = np.full(size, -1, np.int16)
            for row in rows:
                a, b = round(row["start"] * 100), round(row["end"] * 100)
                region = labels[a:b]
                label = names.index(row["speaker_id"]) if row["speaker_id"] in names else -3
                labels[a:b] = np.where(region == -1, label, -2)
            return labels

        r, h = raster(ref, ref_names), raster(hyp, hyp_names)
        valid = (r != -2) & (h != -2)
        for row in ref:
            for at in (row["start"], row["end"]):
                valid[max(0, round((at - collar) * 100)):round((at + collar) * 100)] = False
        scored += int(valid.sum())
        ref_speech, hyp_speech = r >= 0, h != -1
        speech += int(np.count_nonzero(valid & ref_speech))
        missed += int(np.count_nonzero(valid & ref_speech & ~hyp_speech))
        false_alarm += int(np.count_nonzero(valid & ~ref_speech & hyp_speech))
        weights = [[int(np.count_nonzero(valid & (r == i) & (h == j)))
                    for j in range(len(hyp_names))] for i in range(len(ref_names))]

        @lru_cache(None)
        def matching(i, used):
            if i == len(ref_names):
                return 0
            return max([matching(i + 1, used), *[
                weights[i][j] + matching(i + 1, used | (1 << j))
                for j in range(len(hyp_names)) if not used & (1 << j)]])

        both = int(np.count_nonzero(valid & ref_speech & hyp_speech))
        confused += both - matching(0, 0)
    return {"wer": errors / words if words else None,
            "word_errors": errors, "reference_words": words,
            "non_overlap_der": (missed + false_alarm + confused) / speech if speech else None,
            "missed_seconds": missed / 100, "false_alarm_seconds": false_alarm / 100,
            "speaker_error_seconds": confused / 100, "reference_speech_seconds": speech / 100,
            "scored_seconds": scored / 100, "collar_seconds": collar,
            "notes": "10 ms grid; excludes overlap in either annotation; optimal speaker mapping per source."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("hypothesis", type=Path)
    parser.add_argument("--collar", type=float, default=.25)
    args = parser.parse_args()
    if args.collar < 0 or not math.isfinite(args.collar):
        parser.error("--collar must be finite and nonnegative")
    print(json.dumps(evaluate(json.loads(args.reference.read_text()),
                              json.loads(args.hypothesis.read_text()), args.collar), indent=2))
