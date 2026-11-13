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
import tempfile
import threading
import time
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import gigaam
    import numpy as np
    from giga_dictate.engine import GigaAMEngine
    from giga_dictate.meeting_session import MeetingSession
    from giga_dictate.meeting_types import AudioFrame, MeetingOptions, MeetingState

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--mode", choices=("online", "room"), default="room")
    parser.add_argument("--audio", type=Path, default=ROOT / "example.wav")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path, default=ROOT / "build/meeting-benchmark.json")
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("--seconds must be positive")
    cache = Path.home() / ".cache/gigaam"
    weights, tokenizer = cache / "v3_e2e_ctc.ckpt", cache / "v3_e2e_ctc_tokenizer.model"
    if not weights.is_file() or not tokenizer.is_file():
        parser.error("Cached GigaAM weights and tokenizer are required; downloads disabled")
    with wave.open(str(args.audio)) as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16000):
            parser.error("Input must be mono PCM16 WAV at 16 kHz")
        sample = np.frombuffer(handle.readframes(handle.getnframes()), "<i2").astype(np.float32) / 32768
    logging.basicConfig(level=logging.WARNING)
    with (patch.object(gigaam, "_download_model", return_value=("v3_e2e_ctc", str(weights))),
          patch.object(gigaam, "_download_tokenizer", return_value=str(tokenizer)),
          patch("urllib.request.urlopen", side_effect=RuntimeError("Offline benchmark")),
          patch("socket.create_connection", side_effect=RuntimeError("Offline benchmark")),
          tempfile.TemporaryDirectory(prefix="giga-meeting-benchmark-") as tmp):
        engine = GigaAMEngine(device=args.device)
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

        manager = MeetingSession(engine, Path(tmp) / "unused-profile.json", notify,
            microphone_factory=ReplayCapture, system_factory=ReplayCapture,
            clock=lambda: timeline[0])
        manager.start(MeetingOptions(args.mode))
        if not ready.wait(20):
            manager.cancel()
            raise SystemExit("Meeting preparation failed")
        started, cpu = time.monotonic(), time.process_time()
        total = args.seconds * 16000
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
        report = {"mode": args.mode, "audio_seconds_per_source": args.seconds,
                  "device": engine.device_str, "wall_seconds": round(elapsed, 3),
                  "cpu_seconds": round(time.process_time() - cpu, 3),
                  "real_time_factor": round(elapsed / args.seconds, 4),
                  "turn_count": len(manager.result.turns),
                  "speaker_count": len({turn.effective_speaker_id for turn in manager.result.turns
                                        if turn.effective_speaker_id != "unknown"}),
                  "unattributed_turn_count": sum(turn.effective_speaker_id == "unknown"
                                                 for turn in manager.result.turns),
                  "max_queued_seconds": manager.max_queued_seconds,
                  "error_count": len(manager.result.errors),
                  "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "notes": "Accelerated replay of repeated file audio; no live capture, quality or energy measurement."}
        manager.result.clear()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
