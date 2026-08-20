# FL-02: Prompt Iteration Log

## 1. Task Selection

### Task
Debug Python code

### FL-01 Classification
Collaborate with AI

### Original Reasoning
AI helps find bugs, but I verify the solution.

---

## 2. Debugging Problem

### Code

import numpy as np
import pandas as pd

#from sklearn.model_selection import train_test_split

#from sklearn.preprocessing import StandardScaler

#from sklearn.linear_model import LogisticRegression

#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. CUSTOM DATASET

data = {
    "age": [
        22, 25, 28, 31, 35, 40, 45, 50, 52, 58,
        23, 27, 30, 34, 38, 42, 47, 51, 55, 60,
        21, 26, 29, 33, 36, 41, 46, 49, 54, 59
    ],

    "monthly_charges": [
        25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
        28, 32, 38, 42, 48, 52, 58, 62, 68, 72,
        24, 31, 36, 44, 47, 53, 57, 64, 69, 75
    ],

    "support_calls": [
        0, 1, 1, 2, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 1, 2, 3, 3, 4, 5, 6, 8
    ],

    "contract_months": [
        36, 30, 24, 24, 18, 12, 12, 6, 6, 3,
        36, 30, 24, 18, 18, 12, 12, 6, 3, 3,
        36, 30, 24, 24, 18, 12, 12, 6, 6, 3
    ],

    "churn": [
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    ]
}


df = pd.DataFrame(data)


# 2. DISPLAY DATASET

print("\nDataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df["churn"].value_counts())



# 3. PREPARE FEATURES AND TARGET

X = df.drop("churn", axis=1)
y = df["churn"]



# 4. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)



# 5. FEATURE SCALING


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)



# 6. TRAIN MODEL


model = LogisticRegression(
    random_state=42,
    max_iter=100
)

model.fit(X_train_scaled, y_train)


# 7. MAKE PREDICTIONS


y_pred_probability = model.predict_proba(X_test_scaled)[:, 1]

y_pred = (y_pred_probability > 0.7).astype(int)



# 8. EVALUATE MODEL


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# 9. TEST A NEW CUSTOMER

new_customer = pd.DataFrame({
    "age": [45],
    "monthly_charges": [65],
    "support_calls": [5],
    "contract_months": [6]
})


new_customer_scaled = scaler.fit_transform(new_customer)

churn_probability = model.predict_proba(new_customer_scaled)[0][1]

prediction = int(churn_probability > 0.7)


print("\nNew Customer Prediction:")
print("Churn probability:", churn_probability)
print("Prediction:", prediction)

if prediction == 1:
    print("Customer is predicted to churn.")
else:
    print("Customer is predicted t

### Error

there are some intentional bugs in the code.

### Goal

"Given a customer's age, monthly charges, number of support calls, and contract length, will this customer leave the company?"

---

## 3. Version 1 — Naive Prompt

### Technique
Baseline / Naive Prompt

### Prompt
Debug this Python code and tell me what's wrong

### Output
Found several real bugs here — the most serious one will silently give you garbage predictions.

## Bug 1 (critical): Refitting the scaler on test data

```
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)   # ❌ wrong

```

`fit_transform` on `X_test` re-fits the scaler using the test set's own mean/std, instead of applying the scaling parameters learned from training data. This is a classic data leakage / train-test mismatch bug — your test set ends up on a completely different scale than what the model was trained on, which corrupts your evaluation metrics.

**Fix:**

```
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # ✅ transform only, no fit

```

## Bug 2 (critical, worse version of the same mistake): New customer scaling

```
new_customer_scaled = scaler.fit_transform(new_customer)   # ❌ wrong

```

This is the same bug but more damaging: `new_customer` is a single row. Fitting a `StandardScaler` on one sample makes the mean equal to that sample's own values and the standard deviation **zero** for every feature. Dividing by a zero std either throws a warning and produces `0`/`nan`/`inf` values, or (depending on sklearn version) silently produces meaningless scaled values — the input to `model.predict_proba` is essentially garbage, so the churn probability you get for this customer is not trustworthy at all.

**Fix:**

```
new_customer_scaled = scaler.transform(new_customer)   # ✅ reuse the scaler fit on training data

```

## Bug 3 (logic issue, not a crash): Unjustified 0.7 decision threshold

```
y_pred = (y_pred_probability > 0.7).astype(int)
...
prediction = int(churn_probability > 0.7)

```

`LogisticRegression.predict()` by default uses 0.5. Using 0.7 instead means you're only predicting churn when the model is quite confident, which will suppress your recall (you'll miss actual churners) — that's not a bug per se, but with no comment/justification in the code it looks like it was hardcoded by accident rather than a deliberate business decision. Worth either reverting to 0.5 or explicitly documenting *why* 0.7.

## Minor observations (not bugs, but worth knowing)

- The dataset is only 30 rows, split 70/30 → your test set is just 9 samples. Any accuracy/F1 numbers here will be extremely noisy and not statistically meaningful — fine for a toy demo, but don't read much into the specific numbers.
- `max_iter=100` is the sklearn default; it'll likely converge fine on a dataset this small and simple, so not an issue here, but worth bumping if you ever scale up the data and see a `ConvergenceWarning`.

### Corrected snippet (sections 5 and 9)

```
# 5. FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)          # fixed

# 9. TEST A NEW CUSTOMER
new_customer_scaled = scaler.transform(new_customer)  # fixed

```

Once you fix bugs 1 and 2, rerun — your accuracy/classification report numbers will likely change (probably improve, since the model will actually be evaluated fairly), and the new-customer prediction will become meaningful instead of arbitrary.

### What Changed and Why

This was my baseline prompt. I intentionally used a simple one-line instruction without providing a role, additional context, examples, output requirements, or a debugging process. The response successfully identified the major preprocessing issues and also mentioned additional concerns about the decision threshold and the small dataset size
---

## 4. Version 2 — Role Assignment

### Technique
Role Assignment

### Prompt
Act as a senior machine learning engineer with expertise in Python, scikit-learn, and production ML pipelines.

Debug this Python code and tell me what's wrong.

[ EXACT CODE ]

### Output

## Code Review: Churn Prediction Pipeline

Reviewing this as I would a PR — there are two data-leakage bugs that would fail review outright, plus several production-readiness gaps that wouldn't block the PR but I'd flag as follow-ups.

---

### 🔴 Blocking: Test set is being fit, not transformed

```python
X_test_scaled = scaler.fit_transform(X_test)   # line ~103
```

This is textbook train/test contamination. `fit_transform` recomputes mean/std from the test set itself, so `X_test_scaled` lives on a different scale than what the model was trained on. Every downstream metric — accuracy, precision/recall, confusion matrix — is evaluating the model against a distorted input space. In a real pipeline this is the kind of bug that inflates offline metrics and then falls apart in production because inference-time data was never fit on anything.

```python
X_test_scaled = scaler.transform(X_test)   # fit once, on train only
```

### 🔴 Blocking: Inference-time scaler refit on a single row

```python
new_customer_scaled = scaler.fit_transform(new_customer)   # line ~155
```

Same root cause, worse blast radius. This is your inference path — the thing that'll get called in production for every real prediction. Fitting `StandardScaler` on a single row sets std to 0 for every feature, which either throws a divide-by-zero warning and returns 0s, or produces degenerate scaled values depending on your sklearn version. Either way `model.predict_proba` is being fed nonsense. This is the bug that would show up as "model works in notebook, returns garbage in prod" — because your serving code and your training code are silently using different scaling logic.

```python
new_customer_scaled = scaler.transform(new_customer)
```

**Root cause for both:** the scaler is being treated as stateless when it's stateful. This class of bug disappears if you wrap preprocessing + model in a single `sklearn.pipeline.Pipeline` — then there's only one `.fit()` call (on train) and one `.transform()`/`.predict()` path, and it's structurally impossible to accidentally re-fit at inference:

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42, max_iter=1000)),
])
pipe.fit(X_train, y_train)
y_pred_proba = pipe.predict_proba(X_test)[:, 1]
...
churn_probability = pipe.predict_proba(new_customer)[0][1]
```

This also makes the object trivially joblib-serializable as one artifact instead of two things you have to remember to save/load together.

---

### 🟡 Non-blocking, flag for follow-up

**Hardcoded 0.7 threshold, undocumented.**
```python
y_pred = (y_pred_probability > 0.7).astype(int)
```
Default `.predict()` uses 0.5. A 0.7 threshold is a legitimate business decision (e.g., only flag high-confidence churners to avoid alert fatigue on the retention team) — but as written it looks arbitrary. In production I'd want this threshold pulled out as a named constant or config value with a comment explaining the precision/recall tradeoff it encodes, and ideally chosen via a PR curve rather than guessed.

**No cross-validation.** A single 70/30 split on 30 rows means your "accuracy" is one number computed on 9 test samples — it has enormous variance and isn't something I'd trust to compare model versions. Even `StratifiedKFold` with k=5 would give a far more stable signal, small dataset notwithstanding.

**Dataset is synthetic and perfectly separable-looking** (churn correlates almost monotonically with age/charges/support_calls/contract_months). Fine for a demo, but worth calling out explicitly in comments so nobody mistakes these metrics for something transferable to real customer data.

**`max_iter=100`** — works here only because the data is tiny and clean. I'd default new LogisticRegression code to `max_iter=1000` as a habit so it doesn't bite you the first time this gets pointed at a real, larger, messier dataset.

**No `random_state` set on the model's solver behavior beyond what's given**, and no persistence step (`joblib.dump`) — not bugs, but if this is headed toward "production pipeline" as the prompt frames it, model serialization and a `requirements.txt`/pinned sklearn version belong in scope too.

---

### Summary table

| Issue | Severity | Type |
|---|---|---|
| `scaler.fit_transform(X_test)` | Critical | Data leakage |
| `scaler.fit_transform(new_customer)` | Critical | Inference-time leakage (worse) |
| 0.7 threshold undocumented | Medium | Code clarity / process |
| No cross-validation | Medium | Evaluation robustness |
| `max_iter=100` | Low | Latent bug (won't trigger on this data) |
| No `Pipeline` abstraction | Low (root cause of both criticals) | Architecture |

Fix the two critical leakage bugs first — everything else is polish, but those two mean the numbers you're currently looking at aren't measuring what you think they're measuring.

### What Changed and Why

Adding a specific expert role changed the response from a general debugging analysis into a more production-oriented code review. The model still identified the two main scaler-related bugs, but it also evaluated the code from a production ML perspective. It introduced severity levels, discussed inference-time failures, recommended using an sklearn Pipeline, and identified additional concerns such as cross-validation and model serialization.


## 5. Version 3 — Context and Motivation

### Technique
Context and Motivation

### Prompt
Act as a senior machine learning engineer with expertise in Python, scikit-learn, and production ML pipelines.

I am an AI/ML student working on a customer churn prediction project to improve my understanding of machine learning pipelines and debugging.
My goal is not just to get the code fixed, but to understand the root cause of each problem so I can verify the solution myself.

Please debug the following Python code and identify any implementation or machine-learning logic issues that could cause incorrect results or unreliable predictions.


### Output
## Debugging Walkthrough — Churn Prediction Pipeline

Since the goal here is understanding *why* each bug happens (not just patching it), I'll walk through each issue with: what's wrong → why it's wrong → how to verify it yourself → the fix. That way you can reproduce the diagnosis independently, which is really the skill that transfers to your next project.

---

### Issue 1: Test set scaler is `fit_transform`'d instead of `transform`'d

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)   # ← bug
```

**Root cause:** `StandardScaler` is a *stateful* object. Calling `.fit()` computes `mean_` and `scale_` (std) from whatever data you pass it, and `.transform()` applies those stored values. `fit_transform()` is just `fit()` + `transform()` in one call. So the second line throws away the training set's learned mean/std and replaces them with the test set's own mean/std.

**Why this breaks the model, mechanically:** Logistic regression learns coefficients in the coordinate system defined by the training scaler (mean 0, std 1 *for the training data*). If you feed it test data centered around a *different* mean/std, you're evaluating the model on inputs that don't live in the space it learned. It's the same as training a model on Celsius and testing it on Fahrenheit — the model doesn't "know" the units shifted.

**How to verify this yourself:** add these two lines temporarily and look at the numbers:
```python
print(scaler.fit_transform(X_train).mean(axis=0))  # should be ~0
scaler.fit(X_train)
print(scaler.mean_, scaler.scale_)   # note these values
X_test_scaled_buggy = StandardScaler().fit_transform(X_test)
print(StandardScaler().fit(X_test).mean_)  # compare to scaler.mean_ above — they'll differ
```
If `scaler.mean_` differs between the two fits (it will, since train/test are different subsets), you've directly observed the leakage.

**Fix:**
```python
X_test_scaled = scaler.transform(X_test)   # reuse train-fitted scaler, no new fit
```

**Rule to internalize:** `fit`/`fit_transform` happens **once**, on training data only. Every other dataset (validation, test, production input) only ever calls `.transform()`.

---

### Issue 2: Same bug, but on the single-row inference input

```python
new_customer_scaled = scaler.fit_transform(new_customer)   # ← bug
```

**Root cause:** identical mechanism to Issue 1, but with a second-order effect worth understanding: `new_customer` is **one row**. Standard deviation of a single sample is 0 (there's no variance around a mean of one point). `StandardScaler` divides by `scale_`, so you get a `0/0` division for every feature.

**How to verify this yourself:**
```python
test_scaler = StandardScaler()
test_scaler.fit(new_customer)
print(test_scaler.scale_)   # will be array of ~0s (or exactly 0, sklearn substitutes a tiny epsilon)
print(test_scaler.transform(new_customer))  # inspect — likely 0s or extreme/NaN-adjacent values
```
Run that and you'll see `scale_` come back as (effectively) zero — that's the smoking gun. Whatever number comes out of `transform()` after that has no relationship to how "high" or "low" this customer's age/charges/etc. actually are relative to your training population, which is the whole point of scaling.

**Fix:**
```python
new_customer_scaled = scaler.transform(new_customer)
```

**Why this one matters more in practice:** Issue 1 corrupts your *evaluation* — bad, but you'd catch it because your test metrics would look suspicious. Issue 2 corrupts *inference on new data*, which is the actual production use case of this model. A bug here can silently ship wrong predictions with no error thrown, which is the most dangerous kind of ML bug.

---

### Issue 3: Decision threshold of 0.7 instead of the standard 0.5

```python
y_pred = (y_pred_probability > 0.7).astype(int)
prediction = int(churn_probability > 0.7)
```

**Root cause:** this isn't a code bug — it runs fine — but it's an ML *logic* issue if it's unintentional. `LogisticRegression.predict()` uses 0.5 by default because that's the threshold that minimizes total misclassifications under a symmetric loss assumption. Raising it to 0.7 means you only predict "churn" when the model is quite confident, which mechanically **increases precision and decreases recall** — you'll miss more true churners (false negatives) in exchange for fewer false alarms.

**How to verify the effect yourself:** run the classification report at both thresholds and diff the recall for class 1:
```python
for t in [0.5, 0.7]:
    pred = (y_pred_probability > t).astype(int)
    print(t, classification_report(y_test, pred, output_dict=True)['1'])
```
You should see recall for the churn class drop as `t` goes from 0.5 → 0.7.

**Is this "wrong"?** Only you (or the business requirement) can say. If missing a churner is costlier than a false alarm, 0.5 or even lower is more appropriate. If false alarms are costly (e.g., triggering an expensive retention offer), 0.7 could be deliberate. The bug is that it's hardcoded with no comment — so as written, it looks like an accident, not a decision.

---

### Issue 4 (conceptual, not a line-level bug): Evaluation is on 9 samples

30 rows total, 30% test split → your test set is 9 rows. Every metric printed (accuracy, precision, recall) is computed on 9 data points, so a single misclassified row swings accuracy by ~11%. This doesn't need a code fix, but it means you shouldn't draw conclusions like "the model is 89% accurate" from this run — the sample size can't support that claim statistically. Worth knowing conceptually so you don't over-trust metrics on small data going forward (this is a common trap when people first move from toy notebooks to real projects).

---

### Consolidated fix

```python
# Section 5 — fit only on train, transform elsewhere
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)          # was fit_transform

# Section 9 — reuse the already-fitted scaler
new_customer_scaled = scaler.transform(new_customer)  # was fit_transform
```

### What Changed and Why
Adding context about my background and motivation changed the response from primarily a production-oriented code review into a more educational debugging walkthrough.
The model explained not only what was wrong but also why the problems occurred and how I could verify them independently.

Compared with V2, the response placed more emphasis on learning and understanding. For example, it explained the stateful behavior of `StandardScaler`, demonstrated how to verify the scaler problem, explained the precision-recall tradeoff of the 0.7 threshold, and included a self-check exercise. This showed that providing context about my learning goal influenced the depth and teaching style of the response.

## 6. Version 4 — Few-Shot Examples

### Technique
Few-Shot Examples

### Prompt
Act as a senior machine learning engineer with expertise in Python, scikit-learn, and production ML pipelines.

I am an AI/ML student working on a customer churn prediction project to improve my understanding of machine learning pipelines and debugging. My goal is not just to get the code fixed, but to understand the root cause of each problem so I can verify the solution myself.

Here is an example of the debugging style I want:

Example:

Problem:
A model is trained using a StandardScaler fitted on the training data, but the test data is scaled using a newly fitted scaler.

Root cause:
The model was trained in one feature space but evaluated in a different feature space because the test data was normalized using different mean and standard deviation values.

Fix:
Fit the scaler only on the training data and use transform() on the test data.

Verification:
Check that the scaler parameters used for the test data are the same parameters learned from the training data.

Use this style of reasoning when debugging my code. Do not assume that the example describes the bugs in my code; use it only as an example of the level of explanation and verification I want.

Please debug the following Python code and identify any implementation or machine-learning logic issues that could cause incorrect results or unreliable predictions.

[PASTE THE EXACT SAME CODE HERE]

### Output

## Debugging Report — Churn Prediction Pipeline

Following the Problem → Root cause → Fix → Verification structure for each issue found.

### Problem 1
The test set is scaled with `scaler.fit_transform(X_test)` instead of `scaler.transform(X_test)`.

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)   # ← bug
```

**Root cause:**
`StandardScaler` is stateful — `.fit()` computes `mean_` and `scale_` (std) from whatever data is passed to it, and `.transform()` applies those stored values to new data. Calling `fit_transform()` on `X_test` discards the training set's learned mean/std and computes a brand-new mean/std from the test set instead. The model was trained on features centered/scaled according to the *training* distribution, but is then evaluated on features centered/scaled according to the *test* distribution — two different coordinate systems.

**Fix:**
```python
X_test_scaled = scaler.transform(X_test)
```
Fit happens once, on training data only. Every other dataset only ever calls `.transform()`.

**Verification:**
Check that the scaler parameters applied to the test data are the same ones learned from training:
```python
scaler.fit(X_train)
print(scaler.mean_, scaler.scale_)          # the "official" params

leaky_scaler = StandardScaler().fit(X_test)
print(leaky_scaler.mean_, leaky_scaler.scale_)   # compare — these differ
```
If the two printouts differ, that confirms the test data was transformed using the wrong parameters before the fix, and matches after it (since `transform()` no longer refits).

---

### Problem 2
The single-row inference input is scaled with `scaler.fit_transform(new_customer)` instead of `scaler.transform(new_customer)`.

```python
new_customer_scaled = scaler.fit_transform(new_customer)   # ← bug
```

**Root cause:**
Same underlying mechanism as Problem 1, with an additional failure mode: `new_customer` is a single row, so its standard deviation is 0 (there is no spread around a mean computed from one point). `StandardScaler.transform()` divides by `scale_`, so fitting on one row produces a zero (or near-zero, sklearn substitutes a tiny epsilon to avoid a hard crash) denominator. The resulting scaled values bear no relationship to where this customer actually falls relative to the training population — which is the entire purpose of scaling.

**Fix:**
```python
new_customer_scaled = scaler.transform(new_customer)
```

**Verification:**
Fit a throwaway scaler on the single row and inspect its learned std:
```python
probe = StandardScaler().fit(new_customer)
print(probe.scale_)   # ~0 for every feature — confirms the mechanism
```
Then confirm the correct path reuses the training-fitted object:
```python
print(scaler.mean_)   # should be identical before and after this fix — same scaler instance, no new .fit() call
```

---

### Problem 3
Predictions use a hardcoded threshold of 0.7 instead of the standard 0.5, with no justification recorded in the code.

```python
y_pred = (y_pred_probability > 0.7).astype(int)
prediction = int(churn_probability > 0.7)
```

**Root cause:**
`LogisticRegression.predict()` defaults to a 0.5 cutoff on `predict_proba`, which is the threshold that minimizes total misclassifications under equal cost assumptions. Raising the cutoff to 0.7 shifts the precision/recall tradeoff: the model now only labels "churn" when it's substantially confident, which increases precision but lowers recall (more actual churners get missed / classified as false negatives). This isn't a syntax bug — it's a modeling decision that's currently undocumented, so it's indistinguishable from an accidental typo.

**Fix:**
Either revert to the standard threshold, or keep 0.7 but make the decision explicit and justified:
```python
CHURN_DECISION_THRESHOLD = 0.5  # or 0.7, with a comment on why
y_pred = (y_pred_probability > CHURN_DECISION_THRESHOLD).astype(int)
```

**Verification:**
Compare recall on the churn class across thresholds to see the tradeoff directly:
```python
for t in [0.5, 0.7]:
    pred = (y_pred_probability > t).astype(int)
    report = classification_report(y_test, pred, output_dict=True)
    print(t, "recall(churn=1):", report['1']['recall'])
```
Recall for class 1 should drop as the threshold rises from 0.5 to 0.7 — confirming the mechanism, not just asserting it.

---

### Problem 4
The train/test split leaves only 9 rows in the test set (30 total rows × 30%), so every reported metric is computed on a very small sample.

**Root cause:**
This isn't a line of code that's "wrong" — `train_test_split` is behaving correctly — but with `n=30`, a 70/30 split yields a 9-row test set. Each individual misclassified row swings accuracy by roughly 1/9 ≈ 11%. Metrics computed on that few samples carry high variance and shouldn't be interpreted as a reliable estimate of real-world performance.

**Fix:**
Not a code fix so much as an evaluation-methodology change. For a dataset this small, k-fold cross-validation gives a far more stable estimate than a single hold-out split:
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(scores, scores.mean(), scores.std())
```

**Verification:**
Check the spread (`scores.std()`) across folds — a wide spread confirms the single-split accuracy number isn't representative, and directly shows why the original 9-sample metric shouldn't be trusted as a point estimate.

---

### Consolidated fix (sections 5 and 9)

```python
# 5. FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)          # fixed — was fit_transform

# 9. TEST A NEW CUSTOMER
new_customer_scaled = scaler.transform(new_customer)  # fixed — was fit_transform
```

### Self-check before you move on
Before rerunning: predict whether accuracy will go up, down, or become unpredictable once Problems 1 and 2 are fixed, and explain why in terms of feature-space mismatch (not just "because it's fixed now"). Then run it and see if your reasoning matches the outcome — if it doesn't, that's a signal to re-derive Problem 1's mechanism rather than move on to the next task.

### What Changed and Why
Adding a few-shot example made the response more consistently follow the demonstrated debugging pattern.
The model repeatedly used the Problem → Root cause → Fix → Verification structure for each issue. Compared with V3, the response became more consistent in how each problem was analyzed and verified.

The example therefore influenced the model's response style and organization without requiring me to explicitly specify the final output structure.

## 7. Version 5 — Output Structure

### Technique
Output Structure

### Prompt
Act as a senior machine learning engineer with expertise in Python, scikit-learn, and production ML pipelines.

I am an AI/ML student working on a customer churn prediction project to improve my understanding of machine learning pipelines and debugging. My goal is not just to get the code fixed, but to understand the root cause of each problem so I can verify the solution myself.

Here is an example of the debugging style I want:

Example:

Problem:
A model is trained using a StandardScaler fitted on the training data, but the test data is scaled using a newly fitted scaler.

Root cause:
The model was trained in one feature space but evaluated in a different feature space because the test data was normalized using different mean and standard deviation values.

Fix:
Fit the scaler only on the training data and use transform() on the test data.

Verification:
Check that the scaler parameters used for the test data are the same parameters learned from the training data.

Use this style of reasoning when debugging my code. Do not assume that the example describes the bugs in my code; use it only as an example of the level of explanation and verification I want.

Now debug the following Python code.

Return your response using this exact structure:

## 1. Summary
Briefly describe the overall state of the code and identify how many significant issues you found.

## 2. Issues Found

For every issue, use this format:

### Issue [number]: [short name]

**Problem:** What is wrong?

**Root Cause:** Why does the problem occur?

**Impact:** What effect can it have on the model, evaluation, or predictions?

**Fix:** Show the corrected code.

**Verification:** Explain how I can verify that the fix works.

## 3. Additional Observations
List any concerns that are not necessarily bugs but could affect model reliability, maintainability, or production readiness.

## 4. Corrected Code
Provide the relevant corrected sections of the code.

## 5. Verification Checklist
Provide a concise checklist I can use to verify that the debugging fixes were applied correctly.

Do not assume that every unusual design choice is a bug. Clearly distinguish between confirmed bugs, potential issues, and design decisions.

[PASTE THE EXACT SAME CODE HERE]

### Output
## 1. Summary

The pipeline is structurally sound (correct use of `train_test_split` with `stratify`, sensible model choice for a binary classification problem, reasonable evaluation metrics chosen). However, there are **2 confirmed bugs** — both instances of the same underlying mistake (refitting a scaler where it should only transform) — plus **1 design decision that needs justification** and **1 evaluation-methodology concern**. The two confirmed bugs are serious: one corrupts your evaluation metrics, the other corrupts your actual inference output.

---

## 2. Issues Found

### Issue 1: Test set scaled with a newly fitted scaler instead of the training-fitted one

**Problem:**
```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)   # ← bug
```

**Root Cause:**
`StandardScaler` is stateful: `.fit()` computes `mean_` and `scale_` from whatever data is passed to it, and `.transform()` applies those stored values to new data. Calling `fit_transform()` on `X_test` discards the mean/std learned from training and computes an entirely new mean/std from the test set itself. The model's decision boundary was learned in the coordinate space defined by the training data's scaler — feeding it test data scaled to a different mean/std evaluates it in a space it was never trained on.

**Impact:**
Every metric downstream of this — accuracy, precision, recall, confusion matrix — is measuring the model against distorted inputs. The reported performance does not reflect how the model would actually behave on properly scaled data. This can make a good model look bad, a bad model look good, or just produce noisy, non-reproducible numbers.

**Fix:**
```python
X_test_scaled = scaler.transform(X_test)
```

**Verification:**
```python
scaler.fit(X_train)
print(scaler.mean_, scaler.scale_)                    # training params

leaky = StandardScaler().fit(X_test)
print(leaky.mean_, leaky.scale_)                       # compare — will differ
```
If these differ (they will, since train/test are different subsets), that confirms the pre-fix code was evaluating in the wrong feature space. After the fix, `X_test_scaled` should be produced using exactly the `scaler.mean_` / `scaler.scale_` printed above — no second `.fit()` call occurs.

---

### Issue 2: New customer (inference input) scaled with a newly fitted scaler

**Problem:**
```python
new_customer_scaled = scaler.fit_transform(new_customer)   # ← bug
```

**Root Cause:**
Same mechanism as Issue 1, but with a more severe consequence: `new_customer` is a single row. The standard deviation of a single data point is 0 (no variance around a mean computed from one sample). `StandardScaler.transform()` divides by `scale_`, so fitting on one row drives that denominator to zero (sklearn typically substitutes a tiny epsilon rather than raising a hard `ZeroDivisionError`, but the resulting scaled values are still degenerate).

**Impact:**
This is your inference/serving path — the code that represents what would run in production for a real prediction request. The scaled values fed into `model.predict_proba` bear no relationship to where this customer's raw values (age, charges, etc.) actually sit relative to the training population. The resulting `churn_probability` is not a trustworthy number, even though the code runs without error — this is the most dangerous kind of bug because nothing crashes or warns loudly.

**Fix:**
```python
new_customer_scaled = scaler.transform(new_customer)
```

**Verification:**
```python
probe = StandardScaler().fit(new_customer)
print(probe.scale_)     # ~0 for every feature — confirms the degenerate-fit mechanism
```
Then confirm the corrected line reuses the training-fitted scaler instance (no new `.fit()` call), e.g. by checking `scaler.mean_` is unchanged before and after this line executes.

---

### Issue 3 (design decision requiring justification, not a confirmed bug): Hardcoded 0.7 classification threshold

**Problem:**
```python
y_pred = (y_pred_probability > 0.7).astype(int)
prediction = int(churn_probability > 0.7)
```

**Root Cause:**
`LogisticRegression.predict()` defaults to a 0.5 threshold on `predict_proba`. Here the threshold is manually raised to 0.7, with no comment explaining why. This is not inherently wrong — a business case for prioritizing precision over recall (e.g., avoiding false-alarm retention offers) can justify it — but as written it's indistinguishable from an accidental value.

**Impact:**
Raising the threshold from 0.5 to 0.7 increases precision and decreases recall on the churn class — more actual churners will be classified as "will stay" (false negatives). If unintentional, this silently changes what the model is optimized to report.

**Fix (if 0.5 was intended):**
```python
y_pred = (y_pred_probability > 0.5).astype(int)
```
**Fix (if 0.7 is intentional):** keep it, but name and document it:
```python
CHURN_DECISION_THRESHOLD = 0.7  # chosen to prioritize precision; see [ticket/rationale]
y_pred = (y_pred_probability > CHURN_DECISION_THRESHOLD).astype(int)
```

**Verification:**
```python
for t in [0.5, 0.7]:
    pred = (y_pred_probability > t).astype(int)
    r = classification_report(y_test, pred, output_dict=True)
    print(t, "recall(churn=1):", r['1']['recall'], "precision(churn=1):", r['1']['precision'])
```
Recall should drop and precision should rise as the threshold increases from 0.5 to 0.7 — confirms the tradeoff is real and lets you decide which threshold matches your actual goal.

---

## 3. Additional Observations

- **Test set size (9 rows):** With 30 total rows and a 30% split, the test set is only 9 samples. Any single misclassification swings accuracy by ~11%. Not a bug, but the reported metrics shouldn't be treated as statistically reliable — worth switching to `StratifiedKFold` cross-validation for a more stable estimate on data this small.
- **`max_iter=100`:** This is sklearn's default and converges fine on this small, clean dataset — not currently a problem, but a hardcoded low `max_iter` is a common source of silent `ConvergenceWarning`s on larger/messier data. Consider `max_iter=1000` as a defensive default going forward.
- **Synthetic, near-perfectly-separable data:** Churn correlates almost monotonically with all four features. Fine for a learning exercise, but metrics from this dataset won't transfer as intuition to noisier real-world data — worth noting in any write-up so the numbers aren't over-interpreted.
- **No `Pipeline` abstraction:** Not a bug, but structurally, wrapping `StandardScaler` + `LogisticRegression` in a single `sklearn.pipeline.Pipeline` would make Issues 1 and 2 structurally impossible (only one `.fit()` call exists, on training data, and everything else is `.transform()`/`.predict()`). Worth adopting as a habit for future pipelines.
- **No model/scaler persistence:** For a "production pipeline" framing, there's no `joblib.dump`/`load` step to save the fitted scaler + model together, which is usually necessary once this leaves a notebook.

---

## 4. Corrected Code

```python
# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)          # fixed — was fit_transform


# ============================================================
# 9. TEST A NEW CUSTOMER
# ============================================================

new_customer = pd.DataFrame({
    "age": [45],
    "monthly_charges": [65],
    "support_calls": [5],
    "contract_months": [6]
})

new_customer_scaled = scaler.transform(new_customer)   # fixed — was fit_transform

churn_probability = model.predict_proba(new_customer_scaled)[0][1]
prediction = int(churn_probability > 0.7)   # keep or revert to 0.5 — see Issue 3
```

---

## 5. Verification Checklist

- [ ] `X_test_scaled` is produced via `scaler.transform(X_test)`, not `fit_transform`
- [ ] `new_customer_scaled` is produced via `scaler.transform(new_customer)`, not `fit_transform`
- [ ] `scaler.mean_` and `scaler.scale_` are identical whether inspected after scaling `X_train`, `X_test`, or `new_customer` (i.e., the scaler was fit exactly once)
- [ ] Re-run the classification report before/after the fix and confirm the numbers change (leakage was actually affecting output)
- [ ] Decide explicitly whether the 0.7 threshold is intentional; if so, document it as a named constant with rationale
- [ ] (Optional but recommended) Replace the single train/test split with `StratifiedKFold` cross-validation and check the std of scores to gauge metric reliability
- [ ] (Optional, structural) Refactor scaler + model into a single `sklearn.pipeline.Pipeline` to make the fit/transform bug class impossible going forward

### What Changed and Why
We're testing whether explicit output constraints make the response more predictable and easier to evaluate.
---

## 8. Version 6 — Step Decomposition

### Technique
Step Decomposition

### Prompt
Act as a senior machine learning engineer with expertise in Python, scikit-learn, and production ML pipelines.

I am an AI/ML student working on a customer churn prediction project to improve my understanding of machine learning pipelines and debugging. My goal is not just to get the code fixed, but to understand the root cause of each problem so I can verify the solution myself.

Here is an example of the debugging style I want:

Example:

Problem:
A model is trained using a StandardScaler fitted on the training data, but the test data is scaled using a newly fitted scaler.

Root cause:
The model was trained in one feature space but evaluated in a different feature space because the test data was normalized using different mean and standard deviation values.

Fix:
Fit the scaler only on the training data and use transform() on the test data.

Verification:
Check that the scaler parameters used for the test data are the same parameters learned from the training data.

Use this style of reasoning when debugging my code. Do not assume that the example describes the bugs in my code; use it only as an example of the level of explanation and verification I want.

Before producing your answer, analyze the code systematically using these steps:

1. Understand the intended purpose of the complete machine-learning pipeline.
2. Trace the data from the original dataset through preprocessing, training, evaluation, and new-customer prediction.
3. Check whether training, testing, and inference data are processed consistently.
4. Check for data leakage and incorrect use of preprocessing objects.
5. Check the model-training and prediction logic.
6. Check whether the evaluation methodology produces trustworthy metrics.
7. Distinguish confirmed bugs from potential issues and intentional design decisions.
8. For every confirmed bug, determine the root cause, impact, minimal fix, and a way to verify the fix.
9. Review the pipeline for additional reliability or production-readiness concerns.

Return your response using this exact structure:

## 1. Summary

Briefly describe the overall state of the code and identify how many significant issues you found.

## 2. Issues Found

For every issue, use this format:

### Issue [number]: [short name]

**Problem:** What is wrong?

**Root Cause:** Why does the problem occur?

**Impact:** What effect can it have on the model, evaluation, or predictions?

**Fix:** Show the corrected code.

**Verification:** Explain how I can verify that the fix works.

## 3. Additional Observations

List any concerns that are not necessarily bugs but could affect model reliability, maintainability, or production readiness.

## 4. Corrected Code

Provide the relevant corrected sections of the code.

## 5. Verification Checklist

Provide a concise checklist I can use to verify that the debugging fixes were applied correctly.

Do not assume that every unusual design choice is a bug. Clearly distinguish between confirmed bugs, potential issues, and design decisions.

Here is the code to debug:

[PASTE THE EXACT SAME CODE HERE]

### Output
## 1. Summary

Tracing the pipeline end-to-end (raw data → split → scaling → training → evaluation → inference), the overall structure is sound: the split is stratified, the model choice fits a binary classification task, and the evaluation metrics chosen are appropriate for churn prediction. However, tracing how the `scaler` object is used at each stage reveals it is refit twice when it should only be fit once — a data-processing consistency violation that appears in two places. I found **2 confirmed bugs** (same root mechanism, different blast radius), **1 design decision that needs explicit justification**, and **1 evaluation-methodology concern**. No issues were found in the model-training logic itself (`model.fit` is called correctly, once, on training data only).

---

## 2. Issues Found

### Issue 1: Test data scaled with a newly fitted scaler instead of the training-fitted one

**Problem:**
```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)   # ← bug
```

**Root Cause:**
`StandardScaler` is stateful — `.fit()` computes `mean_`/`scale_` from whatever data it's given, `.transform()` applies those stored values to new data. Calling `.fit_transform()` on `X_test` throws away the mean/std learned from `X_train` and computes fresh ones from the test set. Tracing the data flow: the model in step 6 learns its coefficients relative to the *training* scaler's coordinate space, but step 7 evaluates it on data expressed in the *test* scaler's coordinate space — two different feature spaces that happen to share a variable name.

**Impact:**
Every metric computed downstream — accuracy, precision, recall, confusion matrix — reflects the model's performance in a distorted space, not its actual performance. This can make the reported numbers arbitrarily better or worse than reality, and the result isn't reproducible in any meaningful sense (it depends on the test set's own statistics, not the model's true generalization).

**Fix:**
```python
X_test_scaled = scaler.transform(X_test)
```

**Verification:**
```python
scaler.fit(X_train)
print(scaler.mean_, scaler.scale_)                 # training-derived params

leaky = StandardScaler().fit(X_test)
print(leaky.mean_, leaky.scale_)                    # will differ from above
```
Confirming these differ demonstrates the pre-fix code evaluated in the wrong space. After the fix, no second `.fit()` call occurs on `X_test`, so `scaler.mean_`/`scale_` stay fixed at the training-derived values throughout.

---

### Issue 2: New customer (inference input) scaled with a newly fitted scaler

**Problem:**
```python
new_customer_scaled = scaler.fit_transform(new_customer)   # ← bug
```

**Root Cause:**
Identical mechanism to Issue 1, but tracing this specific data path matters: `new_customer` is a single row. Standard deviation of one data point is 0. `StandardScaler.transform()` divides by `scale_`, so fitting on a single row drives that denominator to (effectively) zero, producing degenerate scaled values rather than a meaningful representation of how this customer compares to the training population.

**Impact:**
This is the pipeline's actual inference/serving path — what would run in production per prediction request. Because nothing crashes or throws a visible error, the resulting `churn_probability` looks like a normal float but carries no real signal. This is more dangerous than Issue 1 precisely because it fails silently on the code path that matters most for real usage.

**Fix:**
```python
new_customer_scaled = scaler.transform(new_customer)
```

**Verification:**
```python
probe = StandardScaler().fit(new_customer)
print(probe.scale_)     # ~0 for every feature — confirms the degenerate fit
```
Then confirm the fixed line does not call `.fit()` at all — `scaler.mean_` should be identical before and after this line runs, matching the value fit on `X_train`.

---

### Issue 3 (design decision requiring justification, not a confirmed bug): Hardcoded 0.7 classification threshold

**Problem:**
```python
y_pred = (y_pred_probability > 0.7).astype(int)
prediction = int(churn_probability > 0.7)
```

**Root Cause:**
`LogisticRegression.predict()` defaults to 0.5. Here it's manually raised to 0.7 with no comment or named constant, so — checking the prediction logic in isolation — there's no way to tell whether this is deliberate (e.g., prioritizing precision to avoid wasting retention offers) or a leftover/typo value.

**Impact:**
Raising the threshold from 0.5 → 0.7 trades recall for precision on the churn class: fewer false alarms, but more actual churners misclassified as "will stay" (false negatives). If unintentional, this silently changes the model's practical behavior without changing anything about the model itself.

**Fix (if 0.5 was intended):**
```python
y_pred = (y_pred_probability > 0.5).astype(int)
```
**Fix (if 0.7 is intentional):**
```python
CHURN_DECISION_THRESHOLD = 0.7  # prioritizes precision — document rationale here
y_pred = (y_pred_probability > CHURN_DECISION_THRESHOLD).astype(int)
```

**Verification:**
```python
for t in [0.5, 0.7]:
    pred = (y_pred_probability > t).astype(int)
    r = classification_report(y_test, pred, output_dict=True)
    print(t, "recall:", r['1']['recall'], "precision:", r['1']['precision'])
```
Recall should fall and precision should rise as `t` increases — confirms the tradeoff is real, letting you pick the threshold that matches your actual business goal rather than an arbitrary one.

---

## 3. Additional Observations

- **Test set size (9 rows):** 30 total rows × 30% split → 9 test samples. One misclassified row swings accuracy ~11%. Not a code bug, but the evaluation methodology (step 6 of the analysis) means these metrics shouldn't be trusted as stable estimates. `StratifiedKFold` cross-validation would give a materially more trustworthy signal on data this small.
- **`max_iter=100`:** Default value, converges fine here since the data is small and clean — not currently a bug, but a common latent issue once real-world (larger, noisier) data is substituted in. Defaulting to `max_iter=1000` is a reasonable habit.
- **Synthetic, near-perfectly-separable dataset:** All four features correlate almost monotonically with churn. Fine for a learning exercise, but worth explicitly noting so any reported accuracy isn't mistaken for something that would transfer to messier real data.
- **No `Pipeline` abstraction:** Wrapping `StandardScaler` + `LogisticRegression` in a single `sklearn.pipeline.Pipeline` would make Issues 1 and 2 structurally impossible — only one `.fit()` call would exist (on training data), and every other call would necessarily be `.transform()`/`.predict()`. Worth adopting going forward as a bug-class-elimination habit, not just a fix for this instance.
- **No persistence step:** No `joblib.dump`/`load` for the fitted scaler + model. Not a bug in a notebook context, but relevant if this is meant to model a production pipeline, since scaler and model need to be saved/loaded together to avoid exactly the kind of train/inference mismatch seen in Issue 2.

---

## 4. Corrected Code

```python
# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)          # fixed — was fit_transform


# ============================================================
# 9. TEST A NEW CUSTOMER
# ============================================================

new_customer = pd.DataFrame({
    "age": [45],
    "monthly_charges": [65],
    "support_calls": [5],
    "contract_months": [6]
})

new_customer_scaled = scaler.transform(new_customer)   # fixed — was fit_transform

churn_probability = model.predict_proba(new_customer_scaled)[0][1]
prediction = int(churn_probability > 0.7)   # keep or revert to 0.5 — see Issue 3
```

---

## 5. Verification Checklist

- [ ] `X_test_scaled` comes from `scaler.transform(X_test)`, not `fit_transform`
- [ ] `new_customer_scaled` comes from `scaler.transform(new_customer)`, not `fit_transform`
- [ ] `scaler.mean_` / `scaler.scale_` are identical across every use (train, test, new customer) — confirms a single `.fit()` call
- [ ] Classification report numbers change after the fix (confirms leakage was actually affecting the pre-fix output)
- [ ] 0.7 threshold is either reverted to 0.5 or documented as an intentional, named constant
- [ ] (Optional) Cross-validate with `StratifiedKFold` and inspect score variance to judge how trustworthy the metrics are on this small dataset
- [ ] (Optional, structural) Refactor scaler + model into a single `sklearn.pipeline.Pipeline` to eliminate this bug class permanently


### What Changed and Why

Adding step decomposition gave the model an explicit process for investigating the code before producing the final answer. The prompt instructed it to trace the data through preprocessing, training, evaluation, and inference; check for data leakage; inspect model logic; evaluate the reliability of the metrics; and distinguish confirmed bugs from design decisions.

Compared with V5, the response more explicitly separated confirmed bugs from non-bug concerns. It identified two confirmed scaler-related bugs, treated the 0.7 threshold as a design decision requiring justification, and separately identified the small test set as an evaluation-methodology concern.

The response also provided a verification checklist, making it easier to validate whether the fixes were actually applied. This showed that step decomposition improved the diagnostic process rather than only changing the formatting of the answer.
---

## 9. Overall Iteration Analysis

The six prompt versions progressively increased the amount of guidance given to the model.

### V1 — Naive Prompt

The baseline prompt successfully identified the major scaler-related bugs, but the response was relatively general. It established how well the model could debug the code with minimal instruction.

### V2 — Role Assignment

Adding the senior machine learning engineer role shifted the response toward a production-oriented code review. The response introduced concepts such as PR severity, inference-time failures, production pipelines, and model serialization.

### V3 — Context and Motivation

Adding my background and learning goal changed the response toward a more educational debugging walkthrough. The model focused more heavily on explaining root causes and showing how I could independently verify the issues.

### V4 — Few-Shot Examples

The example demonstrated the desired debugging style. The response consistently followed the Problem → Root Cause → Fix → Verification pattern, showing that examples influenced the organization and reasoning style.

### V5 — Output Structure

Explicit output requirements made the response more predictable and easier to scan. Each issue could be evaluated using consistent fields, making comparison between issues easier.

### V6 — Step Decomposition

The final iteration added an explicit debugging process. The model was instructed to trace the data pipeline, check preprocessing consistency, identify leakage, evaluate the methodology, distinguish bugs from design decisions, and provide verification steps.

### Overall Finding

The progression showed that prompt engineering did not simply make the response longer. Different techniques influenced different aspects of the response. Role assignment influenced the professional perspective, context and motivation influenced the educational focus, few-shot examples influenced reasoning style, output structure influenced consistency, and step decomposition influenced the systematic nature of the analysis.

The final prompt provided the most explicit control over both the debugging process and the resulting output, while the earlier versions demonstrated that even a simple prompt was capable of identifying the most obvious bugs.

---

## 10. Claude vs ChatGPT

Claude identified X and Y and provided a more detailed explanation of the preprocessing problem, while ChatGPT followed the requested five-section structure more strictly. ChatGPT was easier to scan, whereas Claude provided more discussion of production implications.

## 11. Reusable Prompt Template

Instead of:

I am an AI/ML student working on a customer churn prediction project...

create:

I am a [ROLE] working on [PROJECT/DOMAIN].
My goal is [GOAL].

---

## 12. Key Lessons Learned

# Key Lessons Learned

Through this prompt iteration exercise, I learned that prompt engineering is not simply about making a prompt longer. Different prompting techniques influence different aspects of an AI response.

### 1. A simple prompt can still produce useful results

The naive prompt was able to identify the major scaler-related bugs. This showed me that a good model can solve a debugging problem even with minimal instructions. However, the response did not provide the same level of control over perspective, reasoning process, or output format.

### 2. Role assignment changes the perspective of the response

When I assigned the role of a senior machine learning engineer, the response became more production-oriented. It discussed code-review severity, inference-time problems, production pipelines, and maintainability.

This showed me that role assignment can influence the perspective from which the model analyzes a problem.

### 3. Context and motivation can make the response more relevant

When I explained that I was an AI/ML student and wanted to understand the root causes rather than simply receive fixes, the response became more educational.

The model provided more explanations and verification methods, which better matched my learning objective.

### 4. Few-shot examples demonstrate the desired behavior

The few-shot example caused the model to repeatedly follow the demonstrated:

Problem → Root Cause → Fix → Verification

pattern.

This taught me that examples can be more effective than simply describing the desired response because they demonstrate what the expected behavior looks like.

### 5. Explicit output structure improves consistency

When I explicitly specified the required sections, the response became easier to scan and evaluate.

Instead of allowing the model to decide how to organize the debugging report, I defined the structure I wanted. This is particularly useful when the output needs to be compared across multiple iterations or used as a deliverable.

### 6. Step decomposition improves systematic analysis

The final prompt instructed the model to follow a sequence of debugging steps, including tracing the data pipeline, checking preprocessing consistency, checking for leakage, evaluating model logic, and distinguishing bugs from design decisions.

This helped turn the prompt from a simple request into a defined debugging workflow.

### 7. More detailed prompts are not automatically better

One important lesson was that adding instructions does not guarantee a better answer.

Each technique should have a specific purpose. Adding unnecessary instructions can make a prompt longer without improving the result.

The goal should be to add information that reduces ambiguity and guides the model toward the desired outcome.

### 8. AI output still needs verification

Even though the AI identified the important bugs, I should not automatically accept every recommendation.

For example, the 0.7 classification threshold is not necessarily a programming bug. Whether it is appropriate depends on the business objective and the precision-recall tradeoff.

This reinforced the importance of verifying AI-generated technical explanations rather than blindly applying them.

### 9. Prompt engineering is iterative

The most important lesson from this assignment is that effective prompting is an iterative process.

I started with:

"Debug this Python code and tell me what's wrong."

and progressively added:

- role
- context
- motivation
- examples
- output structure
- debugging steps

Each iteration gave me more control over how the model approached and presented the problem.

### 10. The best reusable prompts are task-oriented

A reusable prompt should not depend on my specific customer churn project.

Instead, it should contain general placeholders for:

- the user's role
- the task
- the goal
- relevant context
- desired output
- required process
- the actual input/code

This makes the prompt useful for other debugging or technical-analysis tasks.
