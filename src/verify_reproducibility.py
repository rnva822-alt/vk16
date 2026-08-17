"""Compare a freshly reproduced final run with the cached challenge artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("cat_all", "cat_recent", "lgb_neutral", "lgb_diverse")
SEEDS = (42, 2026, 777)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    cached = np.load(ROOT / "artifacts" / "probs_test.npz")
    reproduced = np.load(ROOT / "artifacts" / "probs_test_reproduced.npz")
    print("component       max|delta|")
    print("--------------------------")
    for name in COMPONENTS:
        cached_mean = np.mean([cached[f"{name}_s{seed}"] for seed in SEEDS], axis=0)
        delta = np.max(np.abs(cached_mean - reproduced[name]))
        print(f"{name:14s} {delta:.12g}")

    generated = ROOT / "submission.csv"
    reference = ROOT / "submissions" / "submission_equal4_m180.csv"
    generated_hash = sha256(generated)
    reference_hash = sha256(reference)
    generated_submission = pd.read_csv(generated)
    reference_submission = pd.read_csv(reference)
    print("\nsubmission sha256")
    print(f"generated: {generated_hash}")
    print(f"reference: {reference_hash}")
    print(f"identical: {generated_hash == reference_hash}")
    print(f"generated positives: {int(generated_submission['is_valid'].sum())}")
    print(f"reference positives: {int(reference_submission['is_valid'].sum())}")
    print(f"labels identical: {generated_submission.equals(reference_submission)}")


if __name__ == "__main__":
    main()
