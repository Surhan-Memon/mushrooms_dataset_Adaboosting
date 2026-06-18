#  Mushroom Classification — AdaBoost & Feature Importance

A machine learning project that classifies mushrooms as **Edible (e)** or **Poisonous (p)**
based on physical characteristics, using AdaBoost and analyzing which features matter most.

---

## Dataset

- **File:** `mushrooms.csv`
- **Size:** 8,124 samples, 23 categorical columns
- **Target:** `class` → `e` (Edible) / `p` (Poisonous)
- **Features:** 22 categorical attributes (cap shape, odor, gill type, color, habitat, etc.)

### Class Balance
e (Edible):    4,208

p (Poisonous): 3,916
Fairly balanced — no major class imbalance issues.

---

## Exploratory Data Analysis

- **Count plot** of class distribution (edible vs poisonous)
- **`df.describe()`** — overview of unique values per categorical feature
- **Bar plot** of unique value counts per feature, sorted ascending
- **Count plot** of `odor` split by class — odor turns out to be a near-perfect predictor

---

## Preprocessing

- Dropped `class` from features → `X`
- **One-hot encoded** all categorical features with `pd.get_dummies(drop_first=True)`
  - Expanded from 22 columns → **95 binary feature columns**
- Split: **85% training / 15% test** (`test_size=0.15, random_state=101`)

---

## Model 1 — AdaBoost with a Single Stump (n_estimators=1)

```python
AdaBoostClassifier(n_estimators=1)
```

A single decision stump — the weakest possible AdaBoost model, used to find
the single most predictive feature.

### Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| e (Edible) | 0.96 | 0.81 | 0.88 | 655 |
| p (Poisonous) | 0.81 | 0.96 | 0.88 | 564 |
| **Accuracy** | | | **0.88** | 1219 |

### Single Most Important Feature
```python
model.feature_importances_.argmax()  # → 22
X.columns[22]                         # → 'odor_n'
```

> With just **one feature** (`odor_n`, "no odor"), the model already achieves **88% accuracy** —
> a strong signal that odor alone is highly predictive of mushroom safety.

---

## Error Rate vs Number of Estimators

```python
for n in range(1, 96):
    model = AdaBoostClassifier(n_estimators=n)
    # fit, predict, track error rate
```

- Tested `n_estimators` from **1 to 95**
- Plotted **Error Rate vs Number of Estimators**
- Error rate drops sharply as more weak learners are added, then stabilizes

---

## Feature Importance Analysis (Full Model)

After fitting with more estimators, **20 features** had non-zero importance:

| Feature | Importance |
|---------|-----------|
| `spore-print-color_r` | **0.1693** |
| `odor_f` | 0.1548 |
| `odor_c` | 0.0876 |
| `odor_p` | 0.0855 |
| `odor_n` | 0.0760 |
| `gill-size_n` | 0.0476 |
| `stalk-surface-above-ring_k` | 0.0415 |
| `habitat_w` | 0.0381 |
| `stalk-color-below-ring_n` | 0.0372 |
| `ring-type_p` | 0.0365 |
| `odor_l` | 0.0330 |
| `population_c` | 0.0312 |
| `population_v` | 0.0299 |
| `spore-print-color_w` | 0.0247 |
| `ring-type_f` | 0.0235 |
| `stalk-shape_t` | 0.0226 |
| `stalk-root_c` | 0.0196 |
| `gill-spacing_w` | 0.0145 |
| `spore-print-color_u` | 0.0141 |
| `spore-print-color_n` | 0.0128 |

Visualized with a **sorted bar plot** (rotated x-axis labels for readability).

>  **`odor`** and **`spore-print-color`** dominate the top features — confirming that
> smell and spore color are the strongest biological indicators of mushroom toxicity.

---

## Libraries Used

- `numpy`, `pandas` — data handling
- `matplotlib`, `seaborn` — visualization (countplot, barplot)
- `scikit-learn`:
  - `AdaBoostClassifier`
  - `train_test_split`
  - `classification_report`, `confusion_matrix`, `ConfusionMatrixDisplay`
  - `accuracy_score`

---

## How to Run

1. Upload `mushrooms.csv` to your Google Colab environment
2. Open the notebook in **Google Colab** or **Jupyter Notebook**
3. Run all cells in order

---

## Key Takeaways

| Step | Detail |
|------|--------|
| Single stump (1 feature) | 88% accuracy using `odor_n` alone |
| Top feature overall | `spore-print-color_r` |
| Most important category | **Odor** (5 of top 20 features) |
| Final feature space | 95 one-hot encoded columns |

> **Lesson:** Even a single weak learner (one decision stump) can achieve strong accuracy
> when one feature is highly discriminative — odor is essentially a biological "tell"
> for mushroom toxicity, which is why AdaBoost picks it first.
