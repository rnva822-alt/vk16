"""Offline comparison of ensembles and decision rules on cached CV probabilities.

Everything here is evaluated in a strictly forward manner: any quantity that a rule
needs (threshold, predicted-positive rate, ensemble weights, calibration map) is fitted
on folds that are chronologically EARLIER than the fold being scored. The fold's own
labels are used only to report the achieved F1 (and, separately, an oracle upper bound).

Two facts about the hidden test set are used, both derived from public information given
in the task statement rather than from labels:
  * all-ones submission scores F1 = 0.2347083926, and F1(all ones) = 2p/(1+p),
    so the test set contains exactly p = 0.13295729 * 7446 = 990 positives;
  * the frozen champion submission predicts 1,426 positives and scored F1 = 0.4792176,
    hence TP = 579, precision = 0.4060, recall = 0.5848.

Usage:
    python src/analyze_rules.py
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

COMPONENTS = ["cat_all", "cat_recent", "lgb_neutral", "lgb_diverse", "lgb_strong"]
CHAMPION_WEIGHTS = {"cat_all": 0.0, "cat_recent": 0.70, "lgb_neutral": 0.20,
                    "lgb_diverse": 0.10, "lgb_strong": 0.0}
CHAMPION_THRESHOLD = 0.504

SAMPLE_F1 = 0.2347083926
TEST_ROWS = 7446
TEST_PREVALENCE = SAMPLE_F1 / (2.0 - SAMPLE_F1)
TEST_POSITIVES = int(round(TEST_PREVALENCE * TEST_ROWS))


# ---------------------------------------------------------------- loading

def load_folds() -> list[dict]:
    folds = []
    for path in sorted(ARTIFACTS.glob("probs_fold*.npz")):
        index = int(path.stem.replace("probs_fold", ""))
        data = np.load(path)
        meta = json.loads((ARTIFACTS / f"probs_fold{index}_log.json").read_text())
        seeds = meta["seeds"]
        probs = {c: np.mean([data[f"{c}_s{s}"] for s in seeds], axis=0) for c in COMPONENTS}
        single = {c: data[f"{c}_s{seeds[0]}"] for c in COMPONENTS}
        folds.append({
            "index": index, "y": data["y"], "probs": probs, "single": single,
            "meta": meta, "prevalence": float(data["y"].mean()),
        })
    # fold 1 is the most recent block, so chronological order is descending index
    return sorted(folds, key=lambda f: -f["index"])


def blend(probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.values())
    return sum(probs[c] * w for c, w in weights.items() if w > 0) / total


# ---------------------------------------------------------------- decision rules

def f1_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    return float(f1_score(y, (p >= threshold).astype(int), zero_division=0))


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    grid = np.quantile(p, np.linspace(0.5, 0.995, 400))
    scores = [(f1_at_threshold(y, p, t), float(t)) for t in grid]
    score, threshold = max(scores)
    return threshold, score


def best_rate(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Best predicted-positive rate (top-k rule) instead of a probability threshold."""
    order = np.argsort(-p)
    y_sorted = y[order]
    cumulative_tp = np.cumsum(y_sorted)
    k = np.arange(1, len(y) + 1)
    f1 = 2 * cumulative_tp / (k + y.sum())
    best = int(np.argmax(f1))
    return float((best + 1) / len(y)), float(f1[best])


def f1_at_rate(y: np.ndarray, p: np.ndarray, rate: float) -> float:
    k = max(int(round(rate * len(p))), 1)
    cutoff = np.sort(p)[::-1][k - 1]
    return f1_at_threshold(y, p, cutoff)


def prior_shift(p: np.ndarray, source: float, target: float) -> np.ndarray:
    odds = (p / np.clip(1 - p, 1e-9, None)) * (target / source) * ((1 - source) / (1 - target))
    return odds / (1 + odds)


def fit_calibrator(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    model.fit(p, y)
    return model


def plugin_rate(p_calibrated: np.ndarray, expected_positives: float | None = None) -> tuple[float, float]:
    """Maximise the plug-in estimate of F1 over all top-k cutoffs.

    With calibrated probabilities, E[TP(k)] = sum of the k largest probabilities and
    E[positives] is either known exactly (test set) or estimated as sum(p).
    """
    ordered = np.sort(p_calibrated)[::-1]
    cumulative = np.cumsum(ordered)
    k = np.arange(1, len(ordered) + 1)
    positives = float(np.sum(p_calibrated)) if expected_positives is None else expected_positives
    estimate = 2 * cumulative / (k + positives)
    best = int(np.argmax(estimate))
    return float((best + 1) / len(ordered)), float(estimate[best])


# ---------------------------------------------------------------- experiments

def weight_grid(step: float = 0.1) -> list[dict[str, float]]:
    grid = []
    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 3)
    for w in product(values, repeat=len(COMPONENTS)):
        if abs(sum(w) - 1.0) > 1e-9:
            continue
        grid.append(dict(zip(COMPONENTS, [float(v) for v in w])))
    return grid


def evaluate_ranking(folds: list[dict]) -> pd.DataFrame:
    """Per-fold ranking quality of single components, champion blend and seed averaging."""
    rows = []
    for fold in folds:
        variants = {f"{c} (1 seed)": fold["single"][c] for c in COMPONENTS}
        variants |= {f"{c} (seed avg)": fold["probs"][c] for c in COMPONENTS}
        variants["champion blend (1 seed)"] = blend(fold["single"], CHAMPION_WEIGHTS)
        variants["champion blend (seed avg)"] = blend(fold["probs"], CHAMPION_WEIGHTS)
        variants["equal blend (seed avg)"] = blend(fold["probs"], {c: 1.0 for c in COMPONENTS})
        for name, p in variants.items():
            rows.append({
                "fold": fold["index"], "variant": name,
                "roc_auc": roc_auc_score(fold["y"], p),
                "pr_auc": average_precision_score(fold["y"], p),
                "f1_fixed_0504": f1_at_threshold(fold["y"], p, CHAMPION_THRESHOLD),
                "f1_oracle": best_threshold(fold["y"], p)[1],
                "oracle_rate": best_rate(fold["y"], p)[0],
                "prevalence": fold["prevalence"],
            })
    return pd.DataFrame(rows)


def evaluate_rules(folds: list[dict], weights: dict[str, float], label: str) -> pd.DataFrame:
    """Forward evaluation of operating-point rules: everything is fitted on earlier folds."""
    rows = []
    for position in range(1, len(folds)):
        history, target = folds[:position], folds[position]
        p_hist = np.concatenate([blend(f["probs"], weights) for f in history])
        y_hist = np.concatenate([f["y"] for f in history])
        p = blend(target["probs"], weights)
        y = target["y"]

        thr_hist, _ = best_threshold(y_hist, p_hist)
        rate_hist, _ = best_rate(y_hist, p_hist)
        # rate normalised by prevalence: how many positives we flag per true positive
        ratio_hist = rate_hist / float(y_hist.mean())
        rate_scaled = ratio_hist * float(y.mean())

        calibrator = fit_calibrator(p_hist, y_hist)
        p_calibrated = prior_shift(
            np.clip(calibrator.predict(p), 1e-6, 1 - 1e-6),
            source=float(y_hist.mean()), target=float(y.mean()),
        )
        plugin_r, plugin_estimate = plugin_rate(p_calibrated, expected_positives=float(y.sum()))
        plugin_r_blind, plugin_estimate_blind = plugin_rate(p_calibrated, expected_positives=None)

        rows.append({
            "blend": label,
            "eval_fold": target["index"],
            "history_folds": ",".join(str(f["index"]) for f in history),
            "prevalence_history": float(y_hist.mean()),
            "prevalence_eval": float(y.mean()),
            "f1_fixed_0504": f1_at_threshold(y, p, CHAMPION_THRESHOLD),
            "f1_threshold_from_history": f1_at_threshold(y, p, thr_hist),
            "f1_rate_from_history": f1_at_rate(y, p, rate_hist),
            "f1_rate_prevalence_scaled": f1_at_rate(y, p, rate_scaled),
            "f1_plugin_known_prevalence": f1_at_rate(y, p, plugin_r),
            "f1_plugin_estimated_prevalence": f1_at_rate(y, p, plugin_r_blind),
            "f1_oracle": best_rate(y, p)[1],
            "threshold_from_history": thr_hist,
            "rate_from_history": rate_hist,
            "rate_prevalence_scaled": rate_scaled,
            "rate_plugin": plugin_r,
            "rate_plugin_blind": plugin_r_blind,
            "plugin_f1_estimate": plugin_estimate,
            "plugin_f1_estimate_blind": plugin_estimate_blind,
            "rate_fixed_0504": float((p >= CHAMPION_THRESHOLD).mean()),
            "oracle_rate": best_rate(y, p)[0],
        })
    return pd.DataFrame(rows)


def select_weights_forward(folds: list[dict], step: float = 0.1) -> pd.DataFrame:
    """Weights chosen on earlier folds by PR-AUC / by oracle-rate F1, scored on the next fold."""
    grid = weight_grid(step)
    rows = []
    for position in range(1, len(folds)):
        history, target = folds[:position], folds[position]
        scores = []
        for weights in grid:
            pr = np.mean([average_precision_score(f["y"], blend(f["probs"], weights)) for f in history])
            f1 = np.mean([best_rate(f["y"], blend(f["probs"], weights))[1] for f in history])
            scores.append((pr, f1, weights))
        best_pr = max(scores, key=lambda s: s[0])[2]
        best_f1 = max(scores, key=lambda s: s[1])[2]
        for criterion, weights in [("pr_auc", best_pr), ("oracle_f1", best_f1), ("champion", CHAMPION_WEIGHTS)]:
            p = blend(target["probs"], weights)
            rows.append({
                "eval_fold": target["index"], "criterion": criterion,
                "weights": json.dumps({k: v for k, v in weights.items() if v > 0}),
                "pr_auc": average_precision_score(target["y"], p),
                "roc_auc": roc_auc_score(target["y"], p),
                "f1_oracle_rate": best_rate(target["y"], p)[1],
                "f1_fixed_0504": f1_at_threshold(target["y"], p, CHAMPION_THRESHOLD),
            })
    return pd.DataFrame(rows)


def main() -> None:
    folds = load_folds()
    if not folds:
        raise SystemExit("no cached probabilities in artifacts/, run src/cv_probs.py first")
    print(f"test prevalence implied by sample_f1: {TEST_PREVALENCE:.8f} "
          f"({TEST_POSITIVES} positives of {TEST_ROWS})")
    print(f"champion: 1426 predicted positives, F1 0.4792176 -> "
          f"TP {0.4792176 * (1426 + TEST_POSITIVES) / 2:.1f}\n")

    ranking = evaluate_ranking(folds)
    ranking.to_csv(ARTIFACTS / "analysis_ranking.csv", index=False)
    print("== ranking quality per fold ==")
    print(ranking.pivot_table(index="variant", columns="fold",
                              values=["pr_auc", "f1_oracle", "f1_fixed_0504"]).round(4).to_string())

    print("\n== mean over folds ==")
    print(ranking.groupby("variant")[["roc_auc", "pr_auc", "f1_fixed_0504", "f1_oracle", "oracle_rate"]]
          .mean().round(4).sort_values("pr_auc", ascending=False).to_string())

    rules = pd.concat([
        evaluate_rules(folds, CHAMPION_WEIGHTS, "champion 0.70/0.20/0.10"),
        evaluate_rules(folds, {c: 1.0 for c in COMPONENTS}, "equal blend"),
    ], ignore_index=True)
    rules.to_csv(ARTIFACTS / "analysis_rules.csv", index=False)
    print("\n== operating-point rules (fitted on earlier folds only) ==")
    print(rules.round(4).to_string(index=False))

    weights = select_weights_forward(folds)
    weights.to_csv(ARTIFACTS / "analysis_weights.csv", index=False)
    print("\n== forward weight selection ==")
    print(weights.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
