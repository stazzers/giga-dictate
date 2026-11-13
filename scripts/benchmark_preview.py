#!/usr/bin/env python3
"""Offline comparison using the repository sample and already cached weights.

Prints timings and counts only. Never records audio, downloads weights, touches
settings, or prints recognized text. --baseline-ref compares against a git ref.
"""
import argparse
import json
import logging
from pathlib import Path
import resource
import subprocess
import sys
import time
import types
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gigaam
import numpy as np
import torch
from giga_dictate.engine import GigaAMEngine


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def baseline_class(ref):
    source = subprocess.check_output(["git", "show", f"{ref}:giga_dictate/engine.py"], cwd=ROOT, text=True)
    module = types.ModuleType("giga_dictate._benchmark_baseline")
    sys.modules[module.__name__] = module
    exec(compile(source, "<baseline-engine>", "exec"), module.__dict__)
    return module.GigaAMEngine


def measure(engine, sample):
    sr = 16000
    audio = np.tile(sample, int(np.ceil(300 * sr / sample.size)))[:300 * sr]
    engine.reset_preview_cache()
    forwards = [0]
    original = engine.model.forward
    def counted(*args, **kwargs):
        forwards[0] += 1
        return original(*args, **kwargs)
    engine.model.forward = counted
    rows, finals = [], {}
    try:
        for duration in (1, 5, 10, 20, 40, 60, 90, 120, 180, 240, 300):
            before = forwards[0]
            t0, cpu0 = time.monotonic(), time.process_time()
            engine.transcribe_preview(audio[:duration * sr])
            sync()
            row = {"audio_s": duration, "preview_ms": round((time.monotonic() - t0) * 1000, 2),
                   "cpu_ms": round((time.process_time() - cpu0) * 1000, 2),
                   "forward_calls": forwards[0] - before}
            if duration in (10, 60, 300):
                t0 = time.monotonic()
                finals[duration] = engine.transcribe(audio[:duration * sr])
                sync()
                row["final_ms"] = round((time.monotonic() - t0) * 1000, 2)
                row["final_chars"] = len(finals[duration])
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        engine.model.forward = original
    return rows, finals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path, default=ROOT / "build/preview-benchmark.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cache = Path.home() / ".cache/gigaam"
    weights = cache / "v3_e2e_ctc.ckpt"
    tokenizer = cache / "v3_e2e_ctc_tokenizer.model"
    if not weights.is_file() or not tokenizer.is_file():
        parser.error("Cached v3_e2e_ctc weights and tokenizer are required; downloads are disabled")
    with wave.open(str(ROOT / "example.wav")) as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            parser.error("example.wav must be mono 16-bit PCM at 16 kHz")
        sample = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    with (patch.object(gigaam, "_download_model", return_value=("v3_e2e_ctc", str(weights))),
          patch.object(gigaam, "_download_tokenizer", return_value=str(tokenizer)),
          patch("urllib.request.urlopen", side_effect=RuntimeError("Benchmark is offline")),
          patch("socket.create_connection", side_effect=RuntimeError("Benchmark is offline"))):
        engine = GigaAMEngine(device=args.device)
        if not engine.load():
            raise SystemExit("Unable to load cached model")
        baseline = baseline_class(args.baseline_ref)(device=engine.device_str)
        baseline.model = engine.model
        baseline._is_loaded = True
        print("baseline", flush=True)
        old_rows, old_finals = measure(baseline, sample)
        print("current", flush=True)
        new_rows, new_finals = measure(engine, sample)
    report = {"baseline_ref": args.baseline_ref, "device": engine.device_str,
              "baseline": old_rows, "current": new_rows,
              "final_text_equal": {str(k): old_finals[k] == new_finals[k] for k in old_finals},
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "notes": "Repeated bundled sample; one process/shared model; RSS is cumulative, not per variant."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Report:", args.output, flush=True)


if __name__ == "__main__":
    main()
