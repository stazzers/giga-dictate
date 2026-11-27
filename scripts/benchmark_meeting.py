#!/usr/bin/env python3
"""Replay file audio through real VAD/CAM++/GigaAM, without recording or networking.

This is an accelerated pipeline benchmark, not a battery or live-capture test.
Outputs aggregate metadata only; audio and transcript are never written.
"""
import argparse
import json
import logging
from pathlib import Path
import resource
import sys
import threading
import time
from contextlib import nullcontext
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import gigaam
    import numpy as np
    import torch
    from giga_dictate.engine import GigaAMEngine
    from giga_dictate.diarization import OnlineDiarizer
    from giga_dictate.meeting_session import MeetingSession
    from giga_dictate.meeting_types import AudioFrame, MeetingOptions, MeetingState

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--mode", choices=("online", "room"), default="room")
    parser.add_argument("--speaker-count", type=int)
    parser.add_argument("--audio", type=Path, default=ROOT / "example.wav")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--model", choices=("v3_e2e_ctc", "v3_e2e_rnnt"), default="v3_e2e_ctc")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--models", type=Path, help="Verified directory containing the meeting models")
    parser.add_argument("--reference", type=Path, help="JSON annotations; replay the supplied WAV exactly once")
    parser.add_argument("--output", type=Path, default=ROOT / "build/meeting-benchmark.json")
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("--seconds must be positive")
    if args.torch_threads is not None and args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    options = MeetingOptions(args.mode, speaker_count=args.speaker_count)
    options.validate()
    refinement = {}
    original_refine = OnlineDiarizer.refine

    def timed_refine(diarizer, turns, evidence):
        started = time.monotonic()
        refinement["evidence_turns"] = len(evidence)
        try:
            return original_refine(diarizer, turns, evidence)
        finally:
            refinement["wall_seconds"] = round(time.monotonic() - started, 4)
            refinement.update(diarizer.refinement_stats)
    cache = Path.home() / ".cache/gigaam"
    weights, tokenizer = cache / f"{args.model}.ckpt", cache / f"{args.model}_tokenizer.model"
    if not weights.is_file() or not tokenizer.is_file():
        parser.error("Cached GigaAM weights and tokenizer are required; downloads disabled")
    with wave.open(str(args.audio)) as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16000):
            parser.error("Input must be mono PCM16 WAV at 16 kHz")
        sample = np.frombuffer(handle.readframes(handle.getnframes()), "<i2").astype(np.float32) / 32768
    if not sample.size:
        parser.error("Input WAV must contain audio")
    logging.basicConfig(level=logging.WARNING)
    thread_override = (patch("giga_dictate.engine.torch_threads", return_value=args.torch_threads)
                       if args.torch_threads is not None else nullcontext())
    with (patch.object(gigaam, "_download_model", return_value=(args.model, str(weights))),
          patch.object(gigaam, "_download_tokenizer", return_value=str(tokenizer)),
          patch("urllib.request.urlopen", side_effect=RuntimeError("Offline benchmark")),
          patch("socket.create_connection", side_effect=RuntimeError("Offline benchmark")),
          patch.object(OnlineDiarizer, "refine", autospec=True, side_effect=timed_refine),
          thread_override):
        engine = GigaAMEngine(model_name=args.model, device=args.device)
        if not engine.load():
            raise SystemExit("Could not load cached GigaAM")
        ready, done = threading.Event(), threading.Event()
        timeline = [100.0]

        class ReplayCapture:
            def __init__(self, *args):
                pass

            def start(self):
                pass

            def stop(self):
                pass

        def notify(event):
            if event.get("state") == MeetingState.RECORDING:
                ready.set()
            if event["kind"] == "complete":
                done.set()

        from giga_dictate.meeting_models import find_models, verify_models

        manager = MeetingSession(engine, notify=notify,
            microphone_factory=ReplayCapture, system_factory=ReplayCapture,
            model_finder=find_models if args.models is None else lambda: verify_models(args.models),
            clock=lambda: timeline[0])
        manager.start(options)
        if not ready.wait(20):
            manager.cancel()
            raise SystemExit("Meeting preparation failed")
        started, cpu = time.monotonic(), time.process_time()
        total = len(sample) if args.reference else args.seconds * 16000
        for sequence, offset in enumerate(range(0, total, 512)):
            if done.is_set():
                break
            # Backpressure is a replay accelerator only. Real capture never waits.
            while max(manager._queued.values(), default=0) > 90 and not done.is_set():
                time.sleep(.01)
            size = min(512, total - offset)
            indices = (np.arange(size) + offset) % len(sample)
            pcm = sample[indices]
            timeline[0] = 100 + (offset + size) / 16000
            for source in (["microphone", "system"] if args.mode == "online" else ["microphone"]):
                manager._enqueue(AudioFrame(source, sequence, 100 + offset / 16000, 16000, pcm.copy()))
        manager.stop()
        while not done.wait(10):
            print("Processing queued speech…", flush=True)
        elapsed = time.monotonic() - started
        audio_seconds = total / 16000
        report = {"mode": args.mode, "audio_seconds_per_source": audio_seconds,
                  "model": args.model, "device_requested": args.device, "device": engine.device_str,
                  "torch_threads": torch.get_num_threads(), "wall_seconds": round(elapsed, 3),
                  "cpu_seconds": round(time.process_time() - cpu, 3),
                  "real_time_factor": round(elapsed / audio_seconds, 4),
                  "turn_count": len(manager.result.turns),
                  "speaker_count": len({turn.effective_speaker_id for turn in manager.result.turns
                                        if turn.effective_speaker_id != "unknown"}),
                  "unattributed_turn_count": sum(turn.effective_speaker_id == "unknown"
                                                 for turn in manager.result.turns),
                  "max_queued_seconds": manager.max_queued_seconds,
                  "error_count": len(manager.result.errors),
                  "refinement": refinement,
                  "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "notes": "Accelerated file replay. Quality metrics require an annotated reference."}
        if args.reference:
            from scripts.evaluate_meeting import evaluate

            reference = json.loads(args.reference.read_text())
            hypothesis = [dict(source=turn.source, start=turn.start, end=turn.end,
                               speaker_id=turn.effective_speaker_id, text=turn.text)
                          for turn in manager.result.turns]
            report["quality"] = evaluate(reference, hypothesis)
        manager.result.clear()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
