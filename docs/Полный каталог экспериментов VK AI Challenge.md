# Полный каталог экспериментов VK AI Challenge

## Назначение документа

Этот файл является единым реестром всех **существенных** подходов, проверенных для бинарной классификации жалоб `is_valid`. Он объединяет исходную кампанию с 67 named temporal scripts, forensic/stop records и последующую execution-series `EXP_EXEC001–EXP_EXEC014`. Технические rerun после исправления пути к artifact, проверки воспроизводимости и purely diagnostic reports не считаются отдельной ML-гипотезой.

> **Итог:** единственный approved artifact — frozen champion `submissions/submission_dual_lgb_h105.csv`, public F1 **0.4792176039**. Ни один другой кандидат не прошёл требование `ΔF1 >= +0.0025` и `ΔAUC >= 0` на обоих защищённых temporal folds.

## 1. Замороженный champion

| Параметр | Значение |
|---|---|
| Submission file | `submissions/submission_dual_lgb_h105.csv` |
| Public leaderboard F1 | **0.4792176039** |
| SHA-256 | `af411d0a7d4a9c49d8a03704bb8d59101b5c5933d23fb67e97df71cb20048054` |
| Архитектура | Recency CatBoost + neutral LightGBM + diverse LightGBM |
| Веса | `0.70 / 0.20 / 0.10` |
| Решающий порог | `0.504` |
| CatBoost | half-life 105 days, depth 7, learning rate 0.03, `l2_leaf_reg=9`, 288 trees, seed 42 |
| Neutral LGB | leaves 48, min child 35, lambda 2, learning rate 0.03, 252 trees, positive weight 1 |
| Diverse LGB | leaves 64, min child 25, lambda 3, learning rate 0.025, 743 trees, positive weight 2 |
| Latest protected F1 | `0.450714` |
| Latest protected AUC | `0.792479` |
| Latest TP / FP / FN / TN | `647 / 1,019 / 558 / 5,885` |
| Test format | 7,446 ordered rows, columns `claim_id,is_valid`, 1,426 positive labels |
| Upload budget | 1 attempt, intentionally preserved |

The integration test rebuilt the file byte-for-byte with the same SHA-256, schema, ID order and binary-label assertions. Reference artifacts: `results/frozen_champion_pipeline_integration_test.json`, `results/final_champion_model_report.md`, `results/champion_configuration_and_leakage_safe_validation.md`.

## 2. Единый validation contract

Все material candidates обучались только на временно предшествующих строках и сравнивались с same-run frozen baseline. CatBoost/LGB categorical handling, count mappings, target encodings, imputation, recency reference and calibration objects fit only on the applicable training block. Acceptance gate:

| Gate element | Условие |
|---|---|
| Fold topology | Chronological expanding split |
| Main comparisons | Penultimate + latest protected folds |
| Threshold | Frozen `0.504`, кроме EXP_EXEC013, где forward threshold выбирался только на предшествующем fold |
| F1 requirement | `ΔF1 >= +0.0025` on both target folds |
| AUC requirement | `ΔAUC >= 0` on both target folds |
| After passing | Rolling-origin stability audit |
| After failing | No test inference, no replacement CSV, no submission |

## 3. Полная карта material hypothesis families

| № | Семейство / способы | Гипотеза | Что было реально проверено | Итог и причина закрытия | Ключевые artifacts |
|---:|---|---|---|---|---|
| 1 | CatBoost core variants | Другой depth/regularization/grow policy/CTR/boosting может лучше переноситься во времени | Depth, learning rate, regularization, grow policy, boosting, quantization, CTR/composite CTR, `has_time`, seeds | Rejected: no repeated temporal gain; `has_time` latest ΔF1 `−0.016146` | `catboost_*_ablation.json`, `catboost_has_time_gate_decision.md` |
| 2 | LightGBM core variants | Другие leaves, sampling, tree type и regularization уменьшат FP | leaves, bagging, DART/GOSS/RF, feature subsampling, linear trees, seed variants | Rejected; neutral/diverse frozen components are strongest stable pair; linear trees latest `−0.005016` | `lgb_*_ablation.json`, `lightgbm_linear_tree_gate_decision.md` |
| 3 | Alternative learners | Независимый architecture signal повысит ensemble diversity | XGBoost, HistGradientBoosting, ExtraTrees, MLP, sparse logistic, Spline-GAM, CategoricalNB | Weaker standalone or no material transferable diversity | `xgboost_*`, `histgradient_*`, `extratrees_*`, `mlp_*` |
| 4 | Ensemble weights | Другие Cat/LGB proportions improve F1 | manual sweeps, dual-LGB variants, nested Optuna weights | Closed: no transferable improvement over frozen `0.7/0.2/0.1` | `closed_optuna_weights_and_uncertainty_stacking_decision.md` |
| 5 | Cost-sensitive / focal loss | Увеличить attention to difficult positive examples | Focal LightGBM and cost-sensitive loss variants | Penultimate +0.002724 but latest `−0.010641`; fails time transfer | `focal_lightgbm_nested_gate_decision.md` |
| 6 | Entity and target context | Past-only owner/content statistics add context | Hierarchical past stats, decayed TE, entity IDs, owner floor, native entity categories | Temporal overfit/no repeated gain | `hierarchical_past_stats_*`, `decayed_entity_te_*`, `entity_context_*` |
| 7 | Claim ID structure | ID hash/prefix/suffix contains target signal | Feasibility AUC screen | Below 0.51 screen; no predictive structure | `claim_id_structure_screen_gate_decision.md` |
| 8 | Geography/profile/platform | Country pairs, profile consistency or platform regime add stable signal | country/IP/profile signatures, platform experts, weighted recency, time batches | No OOT gain; no unused organizer platform raw fields | `profile_*`, `country_pair_*`, `platform3_*`, `platform_signal_novelty_*` |
| 9 | Behaviour and content | Bot/activity geometry captures abuse pattern | Bot geometry, activity composition/ratios, reaction velocity, content calendar, hex-content lexical | Strict ablations negative or failed rolling gate | `bscg_*`, `activity_*`, `reaction_velocity_*`, `content_calendar_*` |
| 10 | Account/content chronology | Relative age/registration relationships are discriminative | content/account chronology, age-at-registration plausibility | Chronology penultimate `−0.008087`; plausibility latest `−0.013710` | `account_content_chronology_gate_decision.md`, `age_registration_plausibility_ablation.json` |
| 11 | Conditional residuals | Claim-type × reason residual context improves boundary separation | Conditional non-target residuals + behavioural interactions | Latest ΔF1 `−0.012130`; promoted overlapping FP/TP boundary cases | `conditional_residual_interactions_gate_decision.md` |
| 12 | Sequences and history | Prior interactions/entities provide longitudinal context | past-only sequence context; reporter/content histories feasibility | Past-only weak; reporter-history unavailable | `past_only_sequence_context_*`, `interaction_history_*` |
| 13 | Density and drift | Relative frequency/density solves temporal composition shift | raw/normalized density, bounded reweighting, stationarity normalization, drift masks | Raw density was time proxy; normalized/density-router gates failed | `stationarity_normalized_density_gate_decision.md`, `nested_optuna_density_router_gate_decision.md` |
| 14 | Unsupervised anomaly | Anomaly score separates unusual invalid claims | Robust Mahalanobis, PCA reconstruction, Isolation Forest feasibility | Mahalanobis latest `−0.013345`; no useful anomaly separation | `global_robust_mahalanobis_gate_decision.md` |
| 15 | Transductive adaptation | Pseudo-label/self-training helps test-shift adaptation | Nested transductive self-training | Penultimate +0.005207, latest `−0.006496`; non-stationary FP | `nested_transductive_self_training_gate_decision.md` |
| 16 | Global specialists | Type/platform/MoE specialist can outperform global model | claim type 6/16, platform-3 and MoE specialists | No stable advantage over base model | `claim_type_specialist_ablation.json`, `claim_type_moe_specialists_ablation.json` |
| 17 | Local FP/FN corrections | 6×6 / 16×6 local model can resolve dominant errors | TP-vs-FP discriminators, specialist substitution, routers, density router | Separation insufficient; overlapping ranking; router latest 0.44647 vs 0.45071 | `regularized_6x6_stacker_gate_decision.md`, `interaction_history_and_6x6_discriminator_gate_decision.md` |
| 18 | Probability stacking | Meta model can combine base probabilities better | raw probability stacking, regularized stackers, component disagreement | Older-fold gains did not transfer; negative late folds/sign changes | `raw_only_stacker_rolling_origin_gate_decision.md`, `nested_segment_stacker_gate_decision.md` |
| 19 | Calibration / fixed post-processing | Probability transformation or forward threshold can remove FP | Platt, isotonic, group calibration, boundary-safe demotion, threshold grids | `0.500/0.510` worse online; λ=0/no gain; forward threshold failed latest | `online_threshold_sweep_summary.md`, `boundary_fp_cluster_forensic_and_postprocessing_decision.md` |
| 20 | Dynamic thresholding | Threshold by type/platform/confidence/prevalence improves F1 | type/platform/confidence/prevalence rules | Post-hoc duplicate of threshold/router family; closed | `platform_signal_novelty_and_prevalence_postprocessing_stop_record.md` |
| 21 | Forensic diagnostics | Boundary attribution reveals safe correction | SHAP, clusters, CatBoost dominance, disagreement, error maps | Diagnostic only: FP and TP overlap; no legal hand rule | `boundary_ensemble_shap_attribution_interpretation.md`, `catboost_boundary_dominance_interpretation.md` |
| 22 | Temporal stability/drift | Find a stable actionable failing segment | OOT stability, type-7 JS/PSI stress tests | Diagnostic only; unstable small support and drift do not justify router/threshold | `champion_out_of_time_stability_report.md`, `type7_covariate_shift_stress_report.md` |
| 23 | External/new sources | Extra sanctioned raw data could add independent signal | Organizer/public/GitHub/Kaggle review | No new allowed raw source/permission found | `external_research_hypothesis_stop_record.md`, `morning_autonomous_stop_record.md` |

## 4. Detailed execution-series EXP_EXEC001–014

The table below contains the exact controlled runs created after the broader campaign registry. Deltas are against a same-run frozen champion on the corresponding temporal fold. A blank gate is a legacy `EXP_EXEC002` record that predates the current formal gate serialization; its negative latest result still rejects it.

| ID | Exact method / changed component | Penultimate F1 (Δ) | Latest F1 (Δ) | Latest AUC Δ | Decision |
|---|---|---:|---:|---:|---|
| EXP_EXEC001 | Frozen champion independent reproduction | 0.429526 (+0.000000) | **0.450714 (+0.000000)** | +0.000000 | Approved baseline reproduction |
| EXP_EXEC002 | Activity composition + bot geometry + profile/country compact feature union | 0.414317 (−0.004754) | 0.436389 (−0.014325) | −0.002385 | Reject: removed more TP than FP |
| EXP_EXEC003 | Strict LGB leaf support / split gain regularization | 0.425571 (+0.006500) | 0.446508 (−0.004206) | −0.000372 | Reject: older-fold gain did not transfer |
| EXP_EXEC004 | Nested OOF probability stacker, frozen 3-probability panel | 0.403587 (−0.015484) | 0.431388 (−0.019326) | +0.000019 | Reject: recall/demotion loss |
| EXP_EXEC005 | Nested model-disagreement signals + global CatBoost correction | 0.422519 (+0.003447) | 0.442980 (−0.007734) | +0.000093 | Reject: raised FP faster than TP |
| EXP_EXEC006 | Soft-LGB components + model-disagreement correction | 0.425015 (+0.005943) | 0.445897 (−0.004817) | +0.000163 | Reject: higher recall but lower precision |
| EXP_EXEC007 | Behaviour × bot compact interactions with stability filter | 0.413685 (−0.005387) | 0.440386 (−0.010328) | −0.001480 | Reject: both precision and recall declined |
| EXP_EXEC008 | Clipped-logit natural-prior OOF logistic stacker | 0.135177 (−0.283895) | 0.166289 (−0.284425) | +0.001953 | Reject: probability-scale collapse at frozen threshold |
| EXP_EXEC009 | Stronger LGB regularization than EXP_EXEC003 | 0.422639 (+0.003568) | 0.443532 (−0.007182) | −0.000843 | Reject: precision fell with nearly flat recall |
| EXP_EXEC010 | OOF stacker with added HistGradientBoosting probability | 0.403733 (−0.015338) | 0.431950 (−0.018764) | +0.000028 | Reject: demoted 90 TP to remove 202 FP |
| EXP_EXEC011 | Relationship/chronology compact interactions with stability filter | 0.420094 (+0.001023) | 0.436790 (−0.013924) | −0.001408 | Reject: stable univariate signal did not transfer to model gain |
| EXP_EXEC012 | Synergy: stable relationship interactions + nested OOF stacker | 0.406432 (−0.012640) | 0.424386 (−0.026328) | −0.001389 | Reject: interaction and stacker errors compounded |
| EXP_EXEC013 | Forward-only threshold selection for frozen champion | 0.423177 (+0.004105) | 0.446731 (−0.003983) | +0.000000 | Reject: forward threshold 0.490 increased recall but lowered precision |
| EXP_EXEC014 | Fixed 5-component blend: champion + small EXP003 regularized LGB probabilities | 0.404518 (−0.014553) | 0.431248 (−0.019467) | +0.000655 | Reject: high precision but severe recall loss |

### 4.1 EXP_EXEC002 — compact non-target feature union

**Hypothesis.** Activity composition, bot-score geometry and profile/country consistency add complementary deterministic row-wise evidence.

**Changed features.** Activity totals/ratios, `bscg_min/max/ratio/absdiff`, profile/country consistency transforms.

**Measured failure.** Latest fold changed TP `647 → 614` (−33), FP `1,019 → 995` (−24), FN `558 → 591` (+33), TN `5,885 → 5,909` (+24). The feature union became more conservative but removed more true positives than false positives. Tree forensic found broad usage of `bscg_ratio`, `bscg_absdiff`, `acc_likes_per_social` and `acc_total`; these were correlated re-expressions of existing signal, not a stable TP-vs-FP filter.

**Artifacts.** `experiments/exp_exec002/config.json`, `run.py`, `metrics.json`, `f1_drop_analysis.md`, `tree_forensic_interpretation.md`.

### 4.2 EXP_EXEC003 and EXP_EXEC009 — LGB regularization

**Hypothesis.** Larger leaf support, shallower trees, split-gain penalties and L1/L2 shrinkage prevent overfit on numeric tails and improve precision.

**EXP_EXEC003 parameters.** 32 leaves, depth 6, min child 90, min split gain 0.1, lambda 10, alpha 0.25, feature fraction 0.75, bagging 0.8, max bin 127. It was the closest execution candidate on latest F1 (`0.446508`) but still below champion.

**EXP_EXEC009 parameters.** More severe regularization: 24 leaves, depth 5, min child 120, Hessian floor 20, split gain 0.3, lambda 20, alpha 0.5, max bin 63. It increased latest positives `1,666 → 1,717`, left recall nearly flat, reduced precision and failed latest F1.

**Conclusion.** Neither regularization profile created a transferable precision gain. The configuration family is rejected; EXP_EXEC003 is a diagnostic near-miss, not a qualified final component.

### 4.3 EXP_EXEC004 / 008 / 010 / 012 — probability stacking

**EXP_EXEC004.** L2 logistic meta-model on nested chronological OOF probabilities of the three frozen components. It reduced latest positives `1,666 → 1,382` and recall `0.536929 → 0.463071`, producing latest F1 `0.431388`.

**EXP_EXEC008.** Strong L2 logit stacker with natural class prior. It learned a large negative intercept around `−2.49`; its probability scale collapsed under frozen threshold `0.504`. Latest positives dropped `1,666 → 118`, TP `647 → 110`, F1 `0.166289`, although AUC rose slightly. It is a scale/calibration failure, not an F1 gain.

**EXP_EXEC010.** Added a regularized HistGradientBoosting probability to the OOF panel with 90% champion anchor. Latest demoted 292 original positives: 90 TP and 202 FP. Precision improved `0.388355 → 0.405386`, but recall fell `0.536929 → 0.462241`; F1 became `0.431950`.

**EXP_EXEC012.** Used inner-stable relationship/chronology interactions in base components before OOF stacking. Interaction base alone already fell to `0.436790` latest; stacking reduced it further to `0.424386`.

**Conclusion.** All tested stackers behaved primarily as demotion mechanisms, eliminating too many valid cases from the `[0.504, 0.650)` positive boundary. No stacker created an operationally useful new TP/FP separation.

**Artifacts.** Each `experiments/EXP_EXEC00{4,8}/` and `EXP_EXEC010/`, `EXP_EXEC012/` directory contains `config.json`, `run.py`, `log.txt`, metrics and validation predictions. Cross-run forensic: `experiments/oof_stacking_failure_interpretation.md` and `oof_stacking_failure_forensic.json`.

### 4.4 EXP_EXEC005 and EXP_EXEC006 — model disagreement corrections

**Hypothesis.** CatBoost–LGB score gap, component spread and diverse-minus-neutral signals identify cases where the ensemble is wrong.

**EXP_EXEC005.** A nested global CatBoost correction received disagreement features plus raw global context. Latest recall increased `0.536929 → 0.557676`, but precision fell `0.388355 → 0.367414`, because predicted positives rose `1,666 → 1,829`. Latest F1 fell `−0.007734`.

**EXP_EXEC006.** Combined soft-regularized LGB probabilities and the disagreement correction. It improved penultimate but latest F1 was `0.445897`, delta `−0.004817`. It retained more positives but did not resolve FP/TP overlap.

### 4.5 EXP_EXEC007 and EXP_EXEC011 — compact interactions with stability filters

**EXP_EXEC007 pool.** Bot-score absolute gap; bot mean × log likes; bot gap × social activity; claim-user bot × log accept rate. Every pool feature passed the univariate inner stability screen on both outer folds, yet latest F1 fell to `0.440386`. This proves stable univariate ranking is not enough for incremental multivariate tree signal.

**EXP_EXEC011 pool.** Registration-year gap × bot mean; content age × claim bot; friendship × bot gap; age-bucket gap × log likes. The filter selected two features on latest: registration gap × bot mean and friendship × bot gap. Precision and recall both fell; latest F1 `0.436790`.

### 4.6 EXP_EXEC013 — forward temporal threshold tuning

**Hypothesis.** A threshold selected only on the immediately preceding validation block transfers forward without changing probabilities.

**Protocol.** Penultimate threshold was chosen on the reference fold (`0.480`); latest threshold was chosen only on penultimate predictions (`0.490`). Thresholds `0.500` and `0.510` were excluded because of prior online evidence.

**Result.** Penultimate F1 rose `+0.004105`; latest F1 fell `−0.003983`. Latest recall increased `0.536929 → 0.558506`, but precision fell `0.388355 → 0.372235`. This closes forward threshold tuning under the available grid and operating threshold contract.

### 4.7 EXP_EXEC014 — fixed final blend and stable predictor audit

**Hypothesis.** Keep CatBoost dominant but replace part of frozen LGB mass with small fixed weights from EXP_EXEC003 regularized LGB components.

**No-search weights.** CatBoost `0.60`, frozen neutral LGB `0.15`, frozen diverse LGB `0.05`, regularized neutral LGB `0.12`, regularized diverse LGB `0.08`.

**Result.** Latest F1 `0.431248` (delta `−0.019467`). It increased precision `0.388355 → 0.414877` but collapsed recall `0.536929 → 0.448963`, reducing positives `1,666 → 1,304`.

**Top ten stable predictors.** A feature was called stable only if its normalized gain was nonzero in all six component-folds: CatBoost, neutral LGB and diverse LGB over penultimate and latest folds.

| Rank | Feature | Mean normalized gain | Minimum normalized gain |
|---:|---|---:|---:|
| 1 | `claim_type_reason` | 0.191465 | 0.043744 |
| 2 | `claim_user_bot_prediction_score` | 0.047355 | 0.039119 |
| 3 | `claim_type_te` | 0.044280 | 0.017072 |
| 4 | `mean_bot_score_by_claim_type` | 0.035278 | 0.002365 |
| 5 | `mean_user_bot_by_claim_type` | 0.027749 | 0.003660 |
| 6 | `bot_score_sum` | 0.027280 | 0.023433 |
| 7 | `content_age_x_claim_type` | 0.022873 | 0.015165 |
| 8 | `content_age_days` | 0.021377 | 0.011871 |
| 9 | `additional_likes_count` | 0.019916 | 0.013038 |
| 10 | `claim_reason_count` | 0.018728 | 0.012544 |

**Artifacts.** `experiments/EXP_EXEC014/feature_importance_long.csv`, `feature_stability_top10.csv`, metrics, logs and both prediction CSVs.

## 5. Diagnostic findings that guided experiments

| Diagnostic | Measured finding | Why it did not become a rule |
|---|---|---|
| Latest confusion | 647 TP, 1,019 FP, 558 FN, 5,885 TN | Shows FP cost but not a separable causal feature |
| Main boundary band | `[0.510, 0.650)`: 1,172 rows, 799 FP and 373 TP | TP and FP overlap materially; blanket demotion loses too many TP |
| Component dominance | CatBoost contributes around 70% by fixed blend arithmetic | Not evidence that CatBoost-specific threshold is justified |
| OOT F1 | 0.405689 → 0.419198 → 0.431494 | Improving chronology, but persistent overprediction gap remains |
| Prevalence gap | Predicted positives ~23.30% vs actual ~14.73% | Does not justify dynamic prevalence threshold after post-processing family failed |
| Type 7 shift | Platform JS divergence high and small positive support | Instability is diagnostic, not a robust specialist trigger |
| Tree feature forensic | Derived bot/activity features got tree gain but lacked stable FP-vs-TP separation | Explains why compact unions and interactions degraded latest F1 |

## 6. Rejected / blocked approaches not permissible to revive as variants

The following are not fresh hypotheses under current data provenance: another threshold sweep, another seed, a dynamic type/platform/confidence threshold, a SHAP-selected feature subset, reweighting blend components, a new local router, further Optuna search over previously tested families, pseudo-label retries, or another interaction ratio from the same raw columns. Each is a post-hoc variation of an already gate-failed information family.

The only legitimate reopening condition is a new organizer-provided raw field available to all participants, a new labeled tranche that is chronologically valid, or written organizer authorization for a named external source. That source must be archived with hash, schema, access time and row-level join provenance before its labels are measured.

## 7. Current operational state

| Item | Status |
|---|---|
| Approved model | Frozen champion only |
| Candidate replacing champion | None |
| Current public score | 0.4792176039 |
| Last protected local F1 | 0.450714 |
| Remaining submission attempt | 1 |
| Test prediction / new submission after failed runs | Not created |
| Safe action | Preserve champion; upload only with explicit user confirmation; monitor organizer-sanctioned new information |

## 8. Artifact index

| Need | Main path |
|---|---|
| Best submission | `submissions/submission_dual_lgb_h105.csv` |
| Byte-identical verification | `results/frozen_champion_preupload_verification.json` |
| Full broad-family inspection | `results/final_ml_hypothesis_inspection_report.md` |
| Execution runs 001–011 summary CSV | `experiments/execution_experiment_summary.csv` |
| EXP_EXEC012 | `experiments/EXP_EXEC012/metrics.json` |
| EXP_EXEC013 | `experiments/EXP_EXEC013/metrics.json` |
| EXP_EXEC014 | `experiments/EXP_EXEC014/metrics.json` |
| All experiment scripts and prediction files | `experiments/EXP_EXEC*/` and `experiments/exp_exec002/` |
| OOF stacking failure forensic | `experiments/oof_stacking_failure_interpretation.md` |
| Top-10 stable feature ranking | `experiments/EXP_EXEC014/feature_stability_top10.csv` |

## 9. Final conclusion

The campaign used strict chronological validation, same-run baseline comparisons, frozen threshold governance and reproducible artifacts. The strongest public result remains the original frozen blend. All broad model, feature, specialist, stacking, calibration and post-processing families have either failed on the latest protected fold, were diagnostic only, duplicate a closed family, or require unavailable / unpermitted external information. The final attempt should not be spent on an unqualified experimental file.
