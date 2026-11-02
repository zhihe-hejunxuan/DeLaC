#!/usr/bin/env python3
"""Score a JSON feature matrix in the classifier's pinned CrysFV environment."""
from __future__ import annotations
import argparse, json, sys
from io import StringIO
import joblib
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True)
args = ap.parse_args()
artifact = joblib.load(args.artifact)
frame = pd.read_json(StringIO(sys.stdin.read()), orient="split")
scores = artifact["model"].predict_proba(frame[artifact["features"]])[:, 1]
sys.stdout.write(json.dumps(scores.tolist()))
