#!/usr/bin/env python3
"""Prepare pinned, verified meeting assets before packaging or offline use."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from giga_dictate.meeting_models import prepare_models

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("build/meeting_models"))
args = parser.parse_args()
prepare_models(args.output)
print("Meeting models prepared and verified.")
