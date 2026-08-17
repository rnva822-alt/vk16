"""F1 as a function of the predicted-positive rate on the temporal CV folds.

Motivation. The organisers publish the score of the all-ones submission
(F1 = 0.2347083926). For an all-ones prediction F1 = 2p / (1 + p), so the hidden test
prevalence is p = 0.2347083926 / (2 - 0.2347083926) = 0.132957, i.e. exactly 990 positive
rows out of 7,446. The number of positives is therefore *known*, which makes a rate rule
("label the top k rows") strictly better specified than a probability threshold: it is
invariant to any monotone miscalibration between training and test distributions.

This script measures, on every temporal fold, F1 as a function of the multiplier
m = k / (number of positives in the fold). The champion submission sits at m = 1.44
(1,426 positives of an implied 990); the folds agree that the optimum is near m = 1.8.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FOLDS = [1, 2, 3, 4]
SEEDS = [42, 2026, 777]
COMPONENTS = ["cat_all", "cat_recent", "lgb_neutral", "lgb_diverse", "lgb_strong"]
MULTIPLIERS = np.round(np.arange(1.0, 2.65, 0.1), 2)

CHAMPION = {"cat_recent": 0.70, "lgb_neutral": 0.20, "lgb_diverse": 0.10}
EQUAL4 = {c: 1.0 for c in COMPONENTS if c != "lgb_strong"}
EQUAL5 = {c: 1.0 for c in COMPONENTS}

VARIANTS = {
    "champion 0.70/0.20/0.10": ("prob", CHAMPION),
    "equal 4 components": ("prob", EQUAL4),
    "equal 5 components": ("prob", EQUAL5),
    "rank-average 4 components": ("rank", EQUAL4),
    "rank-average champion": ("rank", CHAMPION),
}


def seed_average(data, component: str) -> np.ndarray:
    return np.mean([data[f"{component}_s{seed}"] for seed in SEEDS], axis=0)


def blend(data, weights: dict[str, float], kind: str) -> np.ndarray:
    total = sum(weights.values())
    if kind == "prob":
        return sum(seed_average(data, c) * w for c, w in weights.items()) / total
    size = len(data["y"])
    return sum(rankdata(seed_average(data, c)) / size * w for c, w in weights.items()) / total


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    labels = np.zeros(len(scores), dtype=int)
    labels[np.argsort(-scores, kind="stable")[:k]] = 1
    return labels


def main() -> None:
    rows: list[dict] = []
    bands: list[dict] = []
    for name, (kind, weights) in VARIANTS.items():
        for fold in FOLDS:
            data = np.load(ART / f"probs_fold{fold}.npz", allow_pickle=True)
            y = data["y"].astype(int)
            scores = blend(data, weights, kind)
            size = len(y)
            prevalence = y.mean()
            record = {
                "variant": name,
                "fold": fold,
                "prevalence": round(float(prevalence), 4),
                "roc_auc": round(float(roc_auc_score(y, scores)), 4),
                "pr_auc": round(float(average_precision_score(y, scores)), 4),
            }
            for multiplier in MULTIPLIERS:
                k = int(round(prevalence * multiplier * size))
                record[f"m{multiplier:.1f}"] = round(float(f1_score(y, top_k(scores, k))), 4)
            rows.append(record)

            # Marginal precision of the rows a larger k would add. Extending k is
            # profitable exactly while the marginal precision exceeds F1 / 2.
            order = np.argsort(-scores, kind="stable")
            edges = [0.0, 1.44, 1.6, 1.8, 2.0, 2.2]
            for low, high in zip(edges, edges[1:]):
                idx = order[int(round(prevalence * low * size)):int(round(prevalence * high * size))]
                bands.append({
                    "variant": name, "fold": fold, "band": f"{low}-{high}",
                    "rows": len(idx), "precision": round(float(y[idx].mean()), 4),
                })

    curve = pd.DataFrame(rows)
    curve.to_csv(ART / "analysis_rate_curve.csv", index=False)
    pd.DataFrame(bands).to_csv(ART / "analysis_rate_bands.csv", index=False)

    multiplier_columns = [f"m{m:.1f}" for m in MULTIPLIERS]
    mean_curve = curve.groupby("variant")[multiplier_columns].mean().round(4)
    pd.set_option("display.width", 250)
    print("== F1 by predicted-positive multiplier, mean over the 4 temporal folds ==")
    print(mean_curve.to_string())
    print("\nbest multiplier per variant:", mean_curve.idxmax(axis=1).to_dict())
    print("\n== per fold, equal 4 components ==")
    print(curve[curve.variant == "equal 4 components"].set_index("fold")[multiplier_columns].to_string())


if __name__ == "__main__":
    main()
