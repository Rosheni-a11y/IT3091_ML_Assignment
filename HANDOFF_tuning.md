# Hand-off: Hyperparameter Tuning

**Group 2026-AI-05 · IT3091 · DataCo Late-Delivery Risk**
To: Jayasekara C.D. (IT24103494) — advanced models + tuning

---

## What is already done

`02_Model_Training.ipynb` runs end to end and contains:

- 2 reference points (majority class, `Shipping Mode` rule)
- the baseline (Logistic Regression) and 4 alternatives (Decision Tree, Random Forest,
  XGBoost, MLP), all cross-validated
- comparison table, ROC/PR curves, cost-vs-accuracy, feature importance
- threshold analysis and the final test evaluation

**Current results (5-fold CV ROC-AUC on train):**

| Model | CV ROC-AUC | Val ROC-AUC |
|---|---|---|
| Majority class | — | 0.5000 |
| `Shipping Mode` rule (4 cols) | — | 0.7387 |
| Logistic Regression (baseline) | 0.7483 ±0.0041 | 0.7471 |
| Decision Tree | 0.7724 ±0.0034 | 0.7673 |
| **Random Forest** | **0.7746 ±0.0023** | 0.7691 |
| XGBoost | 0.7741 ±0.0032 | 0.7665 |
| Neural Net (MLP) | 0.7507 ±0.0136 | 0.7407 |

**Final (Random Forest) on test: ROC-AUC 0.7801.**

## Your job

Add a tuning section to `02_Model_Training.ipynb`, between §12 (comparison) and
§13 (threshold analysis).

## Read this before you start — it changes what "success" looks like

**The top three models are statistically tied.** The spread between them (0.0022) is smaller
than the fold-to-fold variation within a single model (~0.0030). Section 12 of the notebook
detects this and prints it.

**So expect tuning to gain roughly 0.002–0.008 ROC-AUC.** That is small, and it is the correct
result — Notebooks 1 and 2 both showed the ceiling is a property of the data, not the method
(four `Shipping Mode` columns alone already reach 0.7387 of the 0.7746 achieved by everything).

> **"We tuned it and gained 0.004, which is inside the noise" is a markable finding.**
> Criterion 6 rewards *honest interpretation*. Do not inflate a small gain, and do not keep
> searching until something looks impressive — that is how you overfit the validation set.

If tuning produces a jump above 0.95, that is leakage, not success. Stop and tell the group.

## Which model to tune

**Random Forest.** Not because it tops the table — it does, but only by 0.0005 over XGBoost,
which is meaningless. The real reasons:

- lowest fold-to-fold standard deviation of the top three (0.0023) → most stable
- gives usable feature importances → feeds the secondary lens
- no extra dependency

Tuning XGBoost as well is fine if you have time. Say in the report why you chose what you chose.

## Starting code

Paste this as a new section. It uses the notebook's existing `CV`, `GROUPS` and
`make_pipeline` — do not rebuild them.

```python
from sklearn.model_selection import RandomizedSearchCV

# SEARCH SPACE. Each range is centred on the untuned value, so we are asking
# "is there a better setting nearby?" rather than searching blind.
param_dist = {
    "clf__n_estimators":     [200, 300, 500, 800],
    "clf__max_depth":        [10, 14, 16, 20, None],
    "clf__min_samples_leaf": [5, 10, 20, 40],
    "clf__max_features":     ["sqrt", "log2", 0.3, 0.5],
}

search = RandomizedSearchCV(
    make_pipeline(RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)),
    param_distributions=param_dist,
    n_iter=12,            # 12 candidates x 3 folds = 36 fits, roughly 7 minutes
    cv=CV,                # the order-aware time split - do NOT swap for KFold
    scoring="roc_auc",
    random_state=RANDOM_STATE,
    n_jobs=1,             # the forest already uses all cores; nesting them thrashes
    verbose=2,
    refit=True,
)
search.fit(X_train, y_train, groups=GROUPS)   # groups= is required by our CV class

print("best params:", search.best_params_)
print(f"best CV ROC-AUC : {search.best_score_:.4f}")
print(f"untuned CV      : {cv_scores['4 · Random Forest'].mean():.4f}")
print(f"gain            : {search.best_score_ - cv_scores['4 · Random Forest'].mean():+.4f}")

tuned = search.best_estimator_
train_and_record("7 · Random Forest (tuned)", tuned.named_steps["clf"])
```

Then re-run §15 (final test evaluation) so the reported numbers use the tuned model.

## Three things that will bite you

1. **`groups=GROUPS` is required.** Our CV is a custom `OrderAwareTimeSeriesSplit` that stops a
   single order appearing on both sides of a fold. Without `groups=`, it will fail.
2. **Do not swap `CV` for `KFold`** to make it faster. One order spans up to 5 rows that share a
   label; a random fold puts the answer on both sides. Notebook 1 §8 documents this.
3. **Keep `n_jobs=1` on the search** while the forest uses `n_jobs=-1`. Nesting both oversubscribes
   the CPU and runs *slower*, not faster.

## Runtime

About **7 minutes** with the settings above. Raising `n_iter` to 30 takes ~18 minutes and will
not find much more — the search space is small and the ceiling is in the data.

## What to record in the decision log

- the search space and why those ranges
- the winning parameters
- the gain over untuned, **and whether it exceeds the fold-to-fold standard deviation**
- your verdict: is the tuned model actually better, or just different?

## Running it

```
Kernel: Python 3.13 (ML Assignment)
```

If the kernel picker offers a Python without pandas, it has selected the Microsoft Store stub.
Pick the interpreter at
`C:\Users\<you>\AppData\Local\Programs\Python\Python313\python.exe` instead.

Requires `xgboost`. If it is missing:

```
py -m pip install xgboost
```

The notebook falls back to sklearn's `HistGradientBoostingClassifier` if xgboost is absent, so
it will still run — but the model name in your results table will differ from ours.
