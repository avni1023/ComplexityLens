# ComplexityLens — Rigorous Model Evaluation, Abstention, and Calibration (Day 12)

**Document Version:** 1.0  
**Generated Date:** 2026-11-16  
**Git Commit:** HEAD (Working tree with Day 12 implementation)  
**Dataset Manifest SHA256:** `0b382938897af0f43de4ddd29b0fdf0306d2f33fbe73ad1c7876d7410d973cd4`  
**Fold Assignment Hash:** `c0f993d0cf3d93fd99696b991316b251842004da77341851e44fc3d0859424c5`  
**Primary Evaluation Artifact:** `artifacts/results/day12_evaluation_results.json`  
**Mirrored Artifact:** `artifacts/evaluation/day12_evaluation_results.json`

---

## 1. Title and Mission Overview

**Project Title:**  
*Explainable Hybrid Static-Analysis and ML Estimation of Time/Space Complexity for C++ Programs*

### Critical Scope Clarification
> **Scope Notice:** The current ComplexityLens prototype implements **TIME COMPLEXITY** estimation only. Auxiliary-space complexity analysis is **not implemented** in the current prototype; any discussion of space complexity refers to architectural design specifications or planned future research work.

The primary objective of Day 12 is to execute a rigorous, leakage-safe empirical evaluation comparing:
1. The **Deterministic Static-Only Rule Engine** (grounded in Clang Abstract Syntax Tree and Intermediate Representation extraction).
2. Five **Machine Learning Candidate Baselines** (`majority_class`, `logistic_regression`, `decision_tree`, `random_forest`, `xgboost`).
3. The **Hybrid Decision Gate** combining static confidence with ML out-of-fold probability distributions at a provisional threshold ($\tau = 0.55$).

All evaluations strictly follow zero-leakage protocols: models are retrained in-memory within cross-validation folds, the frozen test partition is held completely inviolate behind programmatic guards, and all statistical comparisons incorporate cluster-level uncertainty.

---

## 2. Dataset and Partitions

The evaluation dataset was constructed under the Day 7 and Day 9 protocols, organized into algorithm families to prevent intra-family data leakage.

- **Total Benchmark Families:** 53 algorithm families
- **Total Development Programs:** 159 C++ programs (exactly 3 implementations per family)
- **Partitions:** 3-Fold Grouped Cross-Validation (split strictly by `family_id`):
  - **Fold 0:** 18 families, 54 programs
  - **Fold 1:** 18 families, 54 programs
  - **Fold 2:** 17 families, 51 programs
- **Frozen Split Manifest:** `data/splits/split_manifest_v1.0.frozen.json`
- **Integrity Checksum:** SHA256 `0b382938897af0f43de4ddd29b0fdf0306d2f33fbe73ad1c7876d7410d973cd4`
- **Fold Assignment Determinism Hash:** `c0f993d0cf3d93fd99696b991316b251842004da77341851e44fc3d0859424c5`

### Test Partition Isolation
All operations are executed via `load_development_dataset()`. Any attempt to access, inspect, or query records from the frozen test partition raises a fatal `TestPartitionAccessError`. Zero test-set examples were accessed during training, tuning, threshold evaluation, or metric generation.

---

## 3. Static-Only Engine Evaluation

The static complexity analysis engine inspects loop bounds, step patterns, nested loop multiplications, and recursion recurrence trees directly from the AST. It requires no statistical training.

### Mandatory Methodological Disclosure
> **Mandatory Disclosure:** *rules developed with dev data visible; not an independent estimate*

Because static analysis rules and patterns were iteratively developed while inspecting representative programs from the development partition, the static engine's performance on the development partition is **not an independent generalization estimate**. It represents an in-distribution diagnostic of rule expressiveness and precision.

### Performance Summary
- **Overall Accuracy:** `0.415094` (41.51%, 66 / 159 correct)
- **Macro-F1:** `0.475992`
- **Coverage:** `0.333333` (33.33%, 53 programs matched deterministic rules; 106 abstained as `UNKNOWN`)
- **Selective Accuracy:** `0.924528` (92.45%, 49 of the 53 accepted predictions were correct)
- **Selective Risk:** `0.075472` (7.55%, only 4 incorrect predictions among accepted rules)
- **UNKNOWN Recall:** `0.809524` (17 of 21 gold `UNKNOWN` programs were correctly abstained)
- **UNKNOWN Precision:** `0.160377` (17 of 106 predicted `UNKNOWN` programs were gold `UNKNOWN`)
- **False Non-UNKNOWN on Gold UNKNOWN:** `4` programs (gold `UNKNOWN` programs where static engine emitted a concrete complexity class)

When the static engine fires a rule, its predictions are remarkably reliable (92.45% precision), but its scope is constrained to known structural patterns, leading to a high abstention rate (66.67% `UNKNOWN`).

---

## 4. Candidate Machine Learning Models

Five machine learning models were evaluated on the canonical 26-feature representations:
1. **Majority Class Baseline:** Predicts the most frequent development class (`ON`, frequency 39/159). Serves as the uninformative lower-bound anchor.
2. **Logistic Regression:** Multinomial logistic regression ($L_2$ penalty, $C=1.0$, `lbfgs` solver, max 1000 iterations). Features scaled using in-fold `StandardScaler`.
3. **Decision Tree:** Scikit-learn `DecisionTreeClassifier` (`criterion="gini"`, `max_depth=5`, `random_state=42`).
4. **Random Forest:** Scikit-learn `RandomForestClassifier` (200 trees, `max_depth=None`, `min_samples_leaf=1`, `random_state=42`, `n_jobs=1`).
5. **XGBoost:** XGBoost `XGBClassifier` (100 estimators, `max_depth=2`, `learning_rate=0.1`, `tree_method="hist"`, `eval_metric="mlogloss"`, `random_state=42`, `n_jobs=1`).

### In-Memory Refitting Protocol
To eliminate any risk of data contamination or artifact leakage:
- Pre-existing fitted artifacts in `artifacts/models/` are **never loaded** during CV evaluation.
- Every candidate model is instantiated fresh and refitted strictly on the training partition of each fold in memory.
- Standard scalers are fitted exclusively on in-fold training data.

### Methodological Bias Label
> **Model Selection Bias Notice:** Random Forest and XGBoost carry the explicit bias label:  
> **`Selection-biased CV estimate`**  
> Their hyperparameter configurations were selected based on validation performance across these same development folds during Day 11 development. Consequently, their cross-validation estimates suffer from hyperparameter optimization bias. In contrast, Majority Class, Logistic Regression, and Decision Tree represent unbiased CV estimates with pre-fixed hyperparameters.

---

## 5. Reproducibility & Day 11 Verification

The Day 12 in-memory refitting pipeline was verified against the stored Day 11 candidate evaluation results to guarantee exact numerical reproducibility:

| Model | Day 11 Reported Acc | Day 12 Refit Acc | Day 11 Reported Macro-F1 | Day 12 Refit Macro-F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | `0.534591` | `0.534591` | `0.549887` | `0.549887` | **EXACT MATCH** |
| **XGBoost** | `0.528302` | `0.528302` | `0.535573` | `0.535573` | **EXACT MATCH** |

The results match to 6 decimal places, confirming complete algorithmic determinism across random seeds, fold construction, and feature extractions.

---

## 6. Comprehensive Classification Metrics

All metrics follow the canonical 8-class taxonomy order: `O1`, `OLOGN`, `ON`, `ONLOGN`, `ON2`, `ON3`, `OEXP`, `UNKNOWN`. Missing classes are handled with documented zero-division contracts (0.0 precision/recall/F1 when denominator is 0).

### Overall Model Comparison Table

| Model / System | Overall Accuracy | Macro-F1 | Coverage | Selective Accuracy | Selective Risk | Bias Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Majority Class** | `0.245283` | `0.049242` | `1.000000` | `0.245283` | `0.754717` | Pre-fixed (Unbiased) |
| **Decision Tree** | `0.433962` | `0.438586` | `0.861635` | `0.489051` | `0.510949` | Pre-fixed (Unbiased) |
| **Static Engine** | `0.415094` | `0.475992` | `0.333333` | `0.924528` | `0.075472` | Dev-visible rules |
| **Logistic Regression** | `0.503145` | `0.490870` | `0.836478` | `0.556391` | `0.443609` | Pre-fixed (Unbiased) |
| **XGBoost** | `0.528302` | `0.535573` | `0.924528` | `0.544218` | `0.455782` | Selection-biased |
| **Random Forest** | `0.534591` | `0.549887` | `0.830189` | `0.590909` | `0.409091` | Selection-biased |
| **Hybrid ($\tau=0.55$)** | **`0.553459`** | **`0.581540`** | `0.704403` | **`0.696429`** | **`0.303571`** | Provisional Threshold |

---

### Per-Class Precision, Recall, and F1 Breakdown

#### 1. Deterministic Static Engine (Rules Developed with Dev Data Visible)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `O1` | 0.7778 | 0.5833 | 0.6667 | 12 |
| `OLOGN` | 1.0000 | 0.1905 | 0.3200 | 21 |
| `ON` | 0.8750 | 0.3590 | 0.5091 | 39 |
| `ONLOGN` | 1.0000 | 0.2000 | 0.3333 | 15 |
| `ON2` | 1.0000 | 0.4242 | 0.5957 | 33 |
| `ON3` | 1.0000 | 0.3333 | 0.5000 | 9 |
| `OEXP` | 1.0000 | 0.4444 | 0.6154 | 9 |
| `UNKNOWN` | 0.1604 | 0.8095 | 0.2677 | 21 |

#### 2. Random Forest Baseline (Selection-Biased)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `O1` | 0.4000 | 0.6667 | 0.5000 | 12 |
| `OLOGN` | 0.2500 | 0.0952 | 0.1379 | 21 |
| `ON` | 0.4839 | 0.3846 | 0.4286 | 39 |
| `ONLOGN` | 0.6667 | 0.4000 | 0.5000 | 15 |
| `ON2` | 0.7073 | 0.8788 | 0.7838 | 33 |
| `ON3` | 0.8182 | 1.0000 | 0.9000 | 9 |
| `OEXP` | 0.7500 | 1.0000 | 0.8571 | 9 |
| `UNKNOWN` | 0.2593 | 0.3333 | 0.2917 | 21 |

#### 3. XGBoost Baseline (Selection-Biased)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `O1` | 0.5294 | 0.7500 | 0.6207 | 12 |
| `OLOGN` | 0.0000 | 0.0000 | 0.0000 | 21 |
| `ON` | 0.4151 | 0.5641 | 0.4783 | 39 |
| `ONLOGN` | 0.7500 | 0.4000 | 0.5217 | 15 |
| `ON2` | 0.6667 | 0.7879 | 0.7222 | 33 |
| `ON3` | 0.8000 | 0.8889 | 0.8421 | 9 |
| `OEXP` | 0.7500 | 1.0000 | 0.8571 | 9 |
| `UNKNOWN` | 0.3333 | 0.1905 | 0.2424 | 21 |

#### 4. Logistic Regression Baseline (Unbiased)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `O1` | 0.2727 | 0.2500 | 0.2609 | 12 |
| `OLOGN` | 0.1818 | 0.0952 | 0.1250 | 21 |
| `ON` | 0.4878 | 0.5128 | 0.5000 | 39 |
| `ONLOGN` | 0.7000 | 0.4667 | 0.5600 | 15 |
| `ON2` | 0.6591 | 0.8788 | 0.7532 | 33 |
| `ON3` | 1.0000 | 0.4444 | 0.6154 | 9 |
| `OEXP` | 0.7500 | 1.0000 | 0.8571 | 9 |
| `UNKNOWN` | 0.2308 | 0.2857 | 0.2553 | 21 |

#### 5. Hybrid System ($\tau=0.55$)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `O1` | 0.4375 | 0.5833 | 0.5000 | 12 |
| `OLOGN` | 0.6667 | 0.1905 | 0.2963 | 21 |
| `ON` | 0.6000 | 0.4615 | 0.5217 | 39 |
| `ONLOGN` | 1.0000 | 0.4000 | 0.5714 | 15 |
| `ON2` | 0.7778 | 0.8485 | 0.8116 | 33 |
| `ON3` | 1.0000 | 0.6667 | 0.8000 | 9 |
| `OEXP` | 0.7500 | 1.0000 | 0.8571 | 9 |
| `UNKNOWN` | 0.2128 | 0.4762 | 0.2941 | 21 |

---

## 7. Abstention & Selective Classification

In production complexity estimation, predicting `UNKNOWN` when uncertain is far preferable to providing a confident but wrong complexity class.

### Abstention and Quality Metrics Table

| Metric | Static Engine | Random Forest | XGBoost | Logistic Reg | Hybrid ($\tau=0.55$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Programs** | 159 | 159 | 159 | 159 | 159 |
| **Accepted Count** | 53 | 132 | 147 | 133 | 112 |
| **Rejected (`UNKNOWN`) Count** | 106 | 27 | 12 | 26 | 47 |
| **Coverage** | `0.333333` | `0.830189` | `0.924528` | `0.836478` | `0.704403` |
| **Selective Accuracy** | **`0.924528`** | `0.590909` | `0.544218` | `0.556391` | **`0.696429`** |
| **Selective Risk** | **`0.075472`** | `0.409091` | `0.455782` | `0.443609` | **`0.303571`** |
| **UNKNOWN Recall** | **`0.809524`** | `0.333333` | `0.190476` | `0.285714` | `0.476190` |
| **UNKNOWN Precision** | `0.160377` | `0.259259` | `0.333333` | `0.230769` | `0.212766` |
| **False Non-UNKNOWN on Gold UNKNOWN** | **4** | 14 | 17 | 15 | **11** |

### Key Observations:
1. **The Static Engine as a Precision Filter:** When the static engine produces a prediction, it is correct 92.45% of the time, failing on only 4 programs in the entire development set. However, it abstains on 66.67% of cases.
2. **ML Models Avoid Abstention:** Scored ML models predict concrete complexity classes aggressively (coverage 83%–92%), which boosts overall dataset coverage but results in high selective risk (41%–46% error rates on accepted predictions).
3. **Hybrid Balances Coverage and Risk:** The hybrid engine increases coverage to 70.44% (accepting 112 programs) while maintaining a selective accuracy of 69.64% (selective risk of 30.36%).

---

## 8. Risk-Coverage Analysis

Risk-coverage curves evaluate how selective risk varies as the model is allowed to abstain on low-confidence predictions.

- **Static Engine:** Represents a single deterministic operating point:
  - $\text{Coverage} = 0.3333$, $\text{Selective Risk} = 0.0755$
- **Scored ML Models:**
  - Random Forest: At full coverage (1.0), risk is $0.4654$. At threshold $\tau=0.7$, coverage drops to $0.4591$ while risk drops to $0.1781$.
  - Logistic Regression: Monotonically decreases risk from $0.4969$ (full coverage) down to $0.1297$ at threshold $0.90$.
- **Empirical Monotonicity:** Across all continuous-probability models, selective risk decreases as the confidence threshold increases, validating that model probability magnitudes correlate meaningfully with empirical prediction correctness.

---

## 9. Probability Calibration

Calibration measures how well predicted probability scores reflect true empirical correctness.

### Calibration Protocol and Metrics Table
- **Binning Strategy:** 5 equal-width bins $[0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]$.
- **Empty Bins:** Formally handled by contributing 0 to ECE without altering total sample weighting.
- **Log-Loss Clipping:** Probabilities clamped to $[\epsilon, 1-\epsilon]$ with $\epsilon = 10^{-15}$.

| Model | 5-Bin ECE | Brier Score | Multiclass Log-Loss | Calibration Status |
| :--- | :---: | :---: | :---: | :--- |
| **Majority Class** | `0.754717` | `1.509434` | `26.067001` | `not performed (sample size)` |
| **Decision Tree** | `0.279621` | `0.886003` | `11.632831` | `not performed (sample size)` |
| **Logistic Regression** | **`0.129659`** | **`0.691563`** | **`1.540689`** | `not performed (sample size)` |
| **XGBoost** | `0.153774` | `0.721514` | `1.610937` | `not performed (sample size)` |
| **Random Forest** | `0.183442` | `0.721447` | `5.770442` | `not performed (sample size)` |

### Methodological Disclosure on Post-Hoc Calibration
> **Mandatory Calibration Statement:**  
> **`Calibration status: not performed (sample size)`**

**Why post-hoc calibration was not applied:**  
Applying post-hoc probability calibration (such as Platt scaling, isotonic regression, or temperature scaling) to a dataset of $N=159$ programs grouped into only 53 families introduces severe risks of empirical overfitting. Partitioning 159 samples across 3 cross-validation folds leaves approximately 106 training programs per fold distributed over 8 taxonomy classes (averaging ~13 samples per class). Fitting non-linear calibration maps on such sparse class supports would fit noise and distort uncertainty bounds. Consequently, raw out-of-fold probabilities are evaluated as-is without post-hoc scaling.

---

## 10. Uncertainty and Bootstrap Analysis

Standard i.i.d. bootstraps fail on grouped benchmark data because programs within the same algorithm family are correlated. We perform a **Seeded Family-Cluster Bootstrap**:
- **Cluster Resampling Unit:** `family_id` (53 unique clusters)
- **Replicates:** 1,000 bootstrap draws with replacement
- **Seed:** `42` (fully deterministic)
- **Confidence Level:** 95% percentile intervals ($2.5^{\text{th}}$ to $97.5^{\text{th}}$ percentiles)

### 95% Confidence Intervals Table

| Model / System | Accuracy Point Estimate | Accuracy 95% CI | Macro-F1 Point Estimate | Macro-F1 95% CI |
| :--- | :---: | :---: | :---: | :---: |
| **Majority Class** | `0.2453` | `[0.1321, 0.3585]` | `0.0492` | `[0.0292, 0.0694]` |
| **Static Engine** | `0.4151` | `[0.3269, 0.5031]` | `0.4760` | `[0.3395, 0.5611]` |
| **Decision Tree** | `0.4340` | `[0.3019, 0.5535]` | `0.4386` | `[0.3132, 0.5254]` |
| **Logistic Regression** | `0.5031` | `[0.3836, 0.6352]` | `0.4909` | `[0.3529, 0.5946]` |
| **XGBoost** | `0.5283` | `[0.3962, 0.6541]` | `0.5356` | `[0.3887, 0.6342]` |
| **Random Forest** | `0.5346` | `[0.4088, 0.6542]` | `0.5499` | `[0.4086, 0.6454]` |
| **Hybrid ($\tau=0.55$)** | **`0.5535`** | **`[0.4403, 0.6667]`** | **`0.5815`** | **`[0.4629, 0.6644]`** |

### Crucial Statistical Disclaimer
> **Mandatory Disclaimer:**  
> **`Small gaps are not meaningful.`**

**Interpretation:**  
Notice the extensive overlap among confidence intervals. Random Forest (`[0.4088, 0.6542]`), XGBoost (`[0.3962, 0.6541]`), and Logistic Regression (`[0.3836, 0.6352]`) have overlapping 95% intervals. The nominal gap between Random Forest (0.5346) and XGBoost (0.5283) is 0.0063 (less than 1 percentage point, equivalent to a single benchmark program). Claims of one ML architecture's definitive superiority over another on this dataset are statistically ungrounded.

---

## 11. Pairwise Model Disagreement across Families

To evaluate model diversity, we compute the number of benchmark families (out of 53) where two systems produce differing predictions on at least one program:

| Model Pair | Differing Families / 53 | Disagreement Rate |
| :--- | :---: | :---: |
| **Static Engine vs. Majority Class** | 52 / 53 | 98.11% |
| **Static Engine vs. Decision Tree** | 50 / 53 | 94.34% |
| **Static Engine vs. XGBoost** | 49 / 53 | 92.45% |
| **Static Engine vs. Logistic Regression** | 48 / 53 | 90.57% |
| **Static Engine vs. Random Forest** | 44 / 53 | 83.02% |
| **Hybrid vs. Static Engine** | 34 / 53 | 64.15% |
| **Hybrid vs. Random Forest** | 21 / 53 | 39.62% |
| **Random Forest vs. XGBoost** | 21 / 53 | 39.62% |
| **Random Forest vs. Logistic Regression** | 20 / 53 | 37.74% |
| **Logistic Regression vs. XGBoost** | 19 / 53 | 35.85% |

### Analysis:
- The ML models agree with each other on ~60%–65% of families.
- The Static Engine differs from ML models on over 83% of families, primarily because the static engine abstains (`UNKNOWN`) whenever loop patterns exceed its rule syntax, whereas ML models predict non-`UNKNOWN` classes.
- This structural divergence is precisely what allows the Hybrid system to combine them effectively.

---

## 12. Hybrid Fusion Evaluation

The hybrid decision function implements the pure `decide_fusion()` gate:
1. If the static engine finds a concrete complexity class ($C \neq \text{UNKNOWN}$), the static result is returned with full symbolic justification.
2. If the static engine abstains ($C = \text{UNKNOWN}$) and the ML model's maximum predicted class probability $P_{\text{max}} \ge \tau$, the ML prediction is accepted.
3. Otherwise, the system abstains with `UNKNOWN` and records the diagnostic rationale.

### Hybrid Evaluation at Provisional Threshold $\tau = 0.55$
- **Overall Accuracy:** `0.553459` (55.35%)
- **Macro-F1:** `0.581540`
- **Coverage:** `0.704403` (112 accepted predictions out of 159)
- **Selective Accuracy:** `0.696429` (69.64% of accepted predictions are correct)
- **Selective Risk:** `0.303571` (30.36% error rate on accepted predictions)
- **Gold UNKNOWN Falsely Accepted:** 11 programs

The hybrid gate achieves the highest point-estimate Macro-F1 (`0.5815`) and highest overall accuracy (`0.5535`) on the development set, lifting the static engine's coverage from 33.3% to 70.4% while mitigating ML error rates through selective abstention.

---

## 13. Threshold Sensitivity Analysis

We evaluate the hybrid gate across thresholds $\tau \in [0.00, 1.00]$ in increments of $0.05$:

| Threshold $\tau$ | Accuracy | Macro-F1 | Coverage | Selective Accuracy | Selective Risk | Gold UNKNOWN False Acceptance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | 0.5849 | 0.6009 | 0.8302 | 0.6515 | 0.3485 | 14 |
| **0.20** | 0.5849 | 0.6009 | 0.8302 | 0.6515 | 0.3485 | 14 |
| **0.35** | 0.5849 | 0.6016 | 0.8239 | 0.6565 | 0.3435 | 14 |
| **0.40** | 0.5849 | 0.6150 | 0.8050 | 0.6719 | 0.3281 | 13 |
| **0.45** | 0.5975 | 0.6301 | 0.7799 | 0.6935 | 0.3065 | 12 |
| **0.50** | 0.5723 | 0.6042 | 0.7484 | 0.6891 | 0.3109 | 11 |
| **0.55 (Provisional)** | **0.5535** | **0.5815** | **0.7044** | **0.6964** | **0.3036** | **11** |
| **0.60** | 0.5346 | 0.5638 | 0.6604 | 0.7048 | 0.2952 | 10 |
| **0.65** | 0.5220 | 0.5527 | 0.5786 | 0.7609 | 0.2391 | 7 |
| **0.70** | 0.4969 | 0.5269 | 0.5283 | 0.7738 | 0.2262 | 6 |
| **0.75** | 0.5031 | 0.5330 | 0.4969 | 0.8101 | 0.1899 | 6 |
| **0.80** | 0.4780 | 0.5117 | 0.4591 | 0.8219 | 0.1781 | 6 |
| **0.85** | 0.4591 | 0.4968 | 0.4277 | 0.8382 | 0.1618 | 5 |
| **0.90** | 0.4465 | 0.4863 | 0.3962 | 0.8730 | 0.1270 | 4 |
| **1.00** | 0.4214 | 0.4663 | 0.3585 | 0.8772 | 0.1228 | 4 |

### Boundary Behavior Analysis:
- **$\tau = 0.00$ (Always Accept ML on Static Abstention):** Coverage is maximized ($83.02\%$), equal to the ML model's coverage. Overall accuracy is $0.5849$, but selective risk is higher ($34.85\%$) and 14 gold `UNKNOWN` programs are falsely classified.
- **$\tau = 1.00$ (Strict Rejection):** Only ML predictions with 100% confidence are accepted. Coverage contracts to $35.85\%$, approaching the static-only engine ($33.33\%$), while selective accuracy rises to $87.72\%$.
- **$\tau = 0.55$ (Provisional Balance):** Strikes a balanced compromise: $70.44\%$ coverage with $<31\%$ selective risk.

---

## 14. Error Analysis and Benchmark Family Breakdown

An analysis of prediction patterns across benchmark families reveals specific failure modes:

### 1. Logarithmic Families (`OLOGN`)
- **Finding:** All models struggle on `OLOGN` (Static Recall: 19.05%, RF Recall: 9.52%, XGBoost Recall: 0.00%).
- *Causal Hypothesis:* Programs implementing binary search, binary lifting, or square-and-multiply loops often lack uniform loop iteration syntax. When loop conditions involve complex pointer arithmetic or non-canonical variable mutation (`low = mid + 1`), both the static AST pattern matcher and count-based ML features fail to distinguish $O(\log N)$ from $O(N)$.

### 2. Multiplicative Steps and Nested Logarithms (`ONLOGN`)
- **Finding:** Static engine achieved 100% precision on `ONLOGN` but only 20% recall.
- *Causal Hypothesis:* Merge sort, heap sort, and fast divide-and-conquer algorithms exhibit recursion splits. The static engine's recursion analyzer correctly identifies clean recurrence patterns ($T(N) = 2T(N/2) + O(N)$), but fails on iterative implementations (e.g., bottom-up merge sort) or non-standard partitioning.

### 3. Quadratic and Polynomial Nested Loops (`ON2`, `ON3`)
- **Finding:** Both ML and Hybrid models perform best on polynomial classes (`ON2` F1: 0.81, `ON3` F1: 0.80).
- *Causal Hypothesis:* Nested loop depth (`max_loop_depth`), loop count, and variable interactions provide strong, unambiguous tabular features that decision trees and random forests exploit with high fidelity.

---

## 15. Threats to Validity and Honest Limitations

To maintain scientific integrity for college project evaluation, we explicitly document the following limitations:

1. **Synthetic & Curated Benchmarks:**  
   The benchmark dataset comprises 159 curated C++ algorithmic problems. Real-world commercial C++ software incorporates metaprogramming, dynamic polymorphism, external I/O, concurrency, and library calls that fall outside the supported subset.
2. **Small Dataset Sample Size ($N=53$ families):**  
   With only 53 families, statistical power is constrained. The 95% confidence intervals are wide (~15–25 percentage points), meaning that differences of 2–3% between models cannot be claimed as significant.
3. **Static Rule Authoring Visibility:**  
   Static engine rules were refined while viewing development instances. The static engine's 41.5% accuracy is a rule-coverage diagnostic, not an unobserved generalization metric.
4. **Model Selection Bias:**  
   The reported configurations for Random Forest and XGBoost were chosen after inspecting Day 11 validation performance, introducing selection bias into their CV estimates.
5. **Auxiliary Space Out of Scope:**  
   As noted, auxiliary space complexity is not implemented.

---

## 16. Practical Recommendations & Next Steps

Based on the Day 12 empirical findings:

1. **Threshold Recommendation for Demonstration:**  
   Maintain the provisional fusion threshold $\tau = 0.55$ for the interactive UI. It provides an optimal balance for live demonstration: recognizing ~70% of user-submitted code while maintaining high explainability and low risk of egregious misclassifications.
2. **Prioritize Explanation Fidelity over Pure Accuracy:**  
   When the static engine fires, its symbolic evidence (AST loop trace, bound variables, recurrence formula) provides invaluable pedagogical explanations that black-box ML cannot replicate.
3. **Preserve Test Set Sanctity:**  
   Keep `data/splits/split_manifest_v1.0.frozen.json` strictly sealed until the final formal presentation evaluation.
