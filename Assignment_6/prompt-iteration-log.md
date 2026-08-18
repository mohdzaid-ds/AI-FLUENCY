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
...

### Output
...

### What Changed and Why
...
---

## 5. Version 3 — Context and Motivation

### Technique
Context and Motivation

### Prompt
...

### Output
...

### What Changed and Why
...

---

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
