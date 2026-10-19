# Evaluation

Generated 2026-10-19 07:55:08 UTC by `python -m fraud_mlops.evaluate` (`make evaluate`). Every number in the README's evaluation section comes from the JSON alongside this file.

Dataset: **6,362,620 transactions**, 8,213 fraudulent (0.1291%).

## Protocol

- **Split by time, on whole steps.** PaySim's `step` is a simulated hour. Train on the earliest hours, validate on the next block, test on the latest. A step is never divided between two splits.
- **Threshold tuned on validation only**, to maximise recall subject to precision >= 95%. Test is scored at that threshold and is never used to choose it.
- **Intervals** are percentile bootstraps over the test rows, 1,000 resamples, 95% level.
- **Seed** 42 throughout.

### Time-based split (primary)

| Split | Rows | Steps | Fraud | Fraud rate |
|---|---:|---|---:|---:|
| train | 3,820,599 | 1–281 | 3,193 | 0.0836% |
| validation | 1,248,498 | 282–354 | 762 | 0.0610% |
| test | 1,293,523 | 355–743 | 4,258 | 0.3292% |

### Random split (secondary)

| Split | Rows | Steps | Fraud | Fraud rate |
|---|---:|---|---:|---:|
| train | 3,817,572 | 1–743 | 4,928 | 0.1291% |
| validation | 1,272,524 | 1–743 | 1,643 | 0.1291% |
| test | 1,272,524 | 1–743 | 1,642 | 0.1290% |

Note how lopsided the time split looks against the row counts: PaySim's transaction volume is heavily front-loaded, so 60% of *rows* is a much smaller share of *hours*. The fraud rate also differs between splits, which is a property of the simulator rather than of any model.

## Results — Time-based split (primary)

| Model | Features | PR-AUC [95% CI] | Recall @95% precision [95% CI] | Precision @ threshold | Recall @ threshold | Threshold |
|---|---:|---|---|---:|---:|---:|
| Rule: large TRANSFER/CASH_OUT † | — | 0.0831 [0.0755, 0.0920] | 0.0% [0.0%, 0.0%] | 19.5% | 11.2% | 4452001.0700 |
| Logistic regression (v1 features) | 7 | 0.6677 [0.6545, 0.6806] | 43.1% [41.2%, 45.2%] | 99.7% | 35.4% | 0.9999 |
| XGBoost (v1 features) | 7 | 0.9523 [0.9478, 0.9563] | 82.8% [81.3%, 84.0%] | 99.4% | 74.4% | 0.9961 |
| XGBoost (v2 features) | 14 | 0.9752 [0.9728, 0.9777] | 83.6% [81.9%, 84.8%] | 99.3% | 74.3% | 0.9984 |

**Recall @95% precision** is the best recall the ranking could reach on test at a 95% precision floor — an upper bound, found with hindsight. **Recall @ threshold** is what the deployed cut-off actually delivers, having been chosen on validation. The gap between the two columns is the cost of not being able to tune on the future.

A rule's threshold is in currency units, not a probability — the rule ranks by amount, so its tuned cut-off is an amount.

† No threshold on the validation set reached 95% precision for this row, so its threshold is a fallback (the highest-precision point available) rather than a calibrated operating point. Read its @-threshold columns as 'the best this could do', not as a deployable configuration.

## Results — Random split (secondary)

| Model | Features | PR-AUC [95% CI] | Recall @95% precision [95% CI] | Precision @ threshold | Recall @ threshold | Threshold |
|---|---:|---|---|---:|---:|---:|
| Rule: large TRANSFER/CASH_OUT † | — | 0.0215 [0.0186, 0.0249] | 0.0% [0.0%, 0.0%] | 7.3% | 8.3% | 6249265.8600 |
| Logistic regression (v1 features) | 7 | 0.5731 [0.5502, 0.5964] | 34.3% [29.6%, 38.2%] | 95.4% | 34.0% | 1.0000 |
| XGBoost (v1 features) | 7 | 0.9603 [0.9535, 0.9662] | 84.8% [82.3%, 86.9%] | 94.6% | 85.1% | 0.9909 |
| XGBoost (v2 features) | 14 | 0.9758 [0.9714, 0.9798] | 88.7% [85.7%, 90.6%] | 95.4% | 88.0% | 0.9973 |

**Recall @95% precision** is the best recall the ranking could reach on test at a 95% precision floor — an upper bound, found with hindsight. **Recall @ threshold** is what the deployed cut-off actually delivers, having been chosen on validation. The gap between the two columns is the cost of not being able to tune on the future.

A rule's threshold is in currency units, not a probability — the rule ranks by amount, so its tuned cut-off is an amount.

† No threshold on the validation set reached 95% precision for this row, so its threshold is a fallback (the highest-precision point available) rather than a calibrated operating point. Read its @-threshold columns as 'the best this could do', not as a deployable configuration.

## Leakage and excluded columns

| Column | Why it is excluded |
|---|---|
| `isFlaggedFraud` | PaySim's own rule-based flag (transfers over 200,000). It is the simulator's output, not an observable input — a model that sees it learns the rule. |
| `nameOrig` | Sender account ID. Used to *group* by when computing history, never as a feature: with 6.3M near-unique values a tree would memorise individuals. |
| `nameDest` | Destination account ID. Same reasoning as `nameOrig`. |

### What the balance columns are worth

PaySim generates fraud by emptying the origin account, so the balance columns do not merely correlate with the label — they encode the simulator's rule. The comparison below is the honest way to show it: the same model, same splits, with and without those columns, against a hand-written rule that uses nothing else. A real deployment has no column this clean, which is why the no-balance row is the better guide to what this system would do on real payments.

| Configuration | PR-AUC [95% CI] | Recall @95% precision [95% CI] |
|---|---|---|
| XGBoost (v2 features) | 0.9752 [0.9728, 0.9777] | 83.6% [81.9%, 84.8%] |
| XGBoost (v2 features, balance columns removed) | 0.2146 [0.2023, 0.2263] | 16.8% [15.5%, 18.2%] |
| Rule: PaySim balance signature | 0.9738 [0.9689, 0.9786] | 97.4% [96.9%, 97.8%] |

The hand-written balance signature — a TRANSFER or CASH_OUT that moves exactly the origin account's whole balance, leaving it at zero — matches **8,024 of 8,213 fraudulent transactions (97.70%)** and **1 of 6,354,407 legitimate ones (0.0000%)** across the whole dataset.

## Environment

- **python:** 3.12.2
- **platform:** Darwin arm64
- **features_source:** data/interim/paysim_with_features.parquet
- **mlflow_store:** file:///Users/pranav-j/Code/fraud-mlops/mlruns_evaluation

Total runtime: 193s.
