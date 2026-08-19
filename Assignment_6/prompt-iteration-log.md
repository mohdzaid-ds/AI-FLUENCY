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
...

### Output
...

### What Changed and Why
...

---

## 7. Version 5 — Output Structure

### Technique
Output Structure

### Prompt
...

### Output
...

### What Changed and Why
...

---

## 8. Version 6 — Step Decomposition

### Technique
Step Decomposition

### Prompt
...

### Output
...

### What Changed and Why
...

---

## 9. Overall Iteration Analysis

### V1 vs V2
...

### V2 vs V3
...

### V3 vs V4
...

### V4 vs V5
...

### V5 vs V6
...

---

## 10. Claude vs ChatGPT

### Claude Output
...

### ChatGPT Output
...

### Comparison
...

---

## 11. Reusable Prompt Template

...

---

## 12. Key Lessons Learned

1. ...
2. ...
3. ...
4. ...
5. ...
