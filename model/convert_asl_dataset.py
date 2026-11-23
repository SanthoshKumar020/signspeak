"""
Convert the ASL Now Fingerspelling dataset (Hugging Face:
sid220/asl-now-fingerspelling, MIT License) into SignSpeak's training
format.

The dataset ships as one JSON file per sample, grouped in per-letter
folders (A/....json, B/....json, ...). Each file holds the 21 MediaPipe
Hand Landmarker landmarks of one real hand sample as {x, y, z} points.

This script runs the (x, y) landmarks through the exact same
gestures.normalize() used at prediction time and writes:

  data/landmarks.csv           label + 42 features, for the signs the app
                               supports (APP_LETTERS below) - consumed by
                               model/train_model.py
  data/landmarks_manifest.csv  row-by-row provenance: which source JSON
                               each landmarks.csv row came from

Usage:
  python model/convert_asl_dataset.py <dataset_dir> [--full-out PATH]

  <dataset_dir>  folder with the per-letter subfolders of JSON files
  --full-out     also write a CSV with ALL letters found in the dataset
                 (not just the app's signs), same 43-column format
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gestures import normalize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
APP_LETTERS = ["A", "B", "C", "D", "E", "F", "I", "L", "Y"]


def load_samples(dataset_dir):
    """Return {letter: [(filename, [(x, y), ...21 points]), ...]} sorted."""
    samples = {}
    for letter in sorted(os.listdir(dataset_dir)):
        folder = os.path.join(dataset_dir, letter)
        if not os.path.isdir(folder) or len(letter) != 1:
            continue
        rows = []
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(folder, fn)) as f:
                raw = json.load(f)
            if len(raw) != 21:
                raise ValueError(f"{letter}/{fn}: expected 21 landmarks, "
                                 f"got {len(raw)}")
            rows.append((fn, [(float(p["x"]), float(p["y"])) for p in raw]))
        if rows:
            samples[letter] = rows
    return samples


def write_csv(path, samples, letters):
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for letter in letters:
            for _fn, pts in samples.get(letter, []):
                feats = normalize(pts)
                w.writerow([letter] + [f"{v:.6f}" for v in feats])
                n += 1
    return n


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: convert_asl_dataset.py <dataset_dir> [--full-out PATH]")
    dataset_dir = sys.argv[1]
    full_out = None
    if "--full-out" in sys.argv:
        full_out = sys.argv[sys.argv.index("--full-out") + 1]

    samples = load_samples(dataset_dir)
    print("Letters found:", {k: len(v) for k, v in sorted(samples.items())})

    os.makedirs(DATA_DIR, exist_ok=True)
    landmarks_path = os.path.join(DATA_DIR, "landmarks.csv")
    n = write_csv(landmarks_path, samples, APP_LETTERS)
    print(f"Wrote {n} rows (app signs {APP_LETTERS}) -> {landmarks_path}")

    manifest_path = os.path.join(DATA_DIR, "landmarks_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "label", "source_file"])
        idx = 0
        for letter in APP_LETTERS:
            for fn, _pts in samples.get(letter, []):
                w.writerow([idx, letter, f"{letter}/{fn}"])
                idx += 1
    print(f"Wrote manifest ({idx} rows) -> {manifest_path}")

    if full_out:
        n_all = write_csv(full_out, samples, sorted(samples.keys()))
        print(f"Wrote {n_all} rows (all letters) -> {full_out}")


if __name__ == "__main__":
    main()
