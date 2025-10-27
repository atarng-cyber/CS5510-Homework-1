# ps2_starter.py
# Starter and utilities for reconstruction attack + defenses.
# No external downloads other than the CSV URL (the script below embeds the CSV).
import numpy as np
import pandas as pd

# --- Dataset: embedded CSV text (exact data you gave) ---
csv_text = """age,sex,blood,admission,result
30,1,3,1,0
62,1,0,2,0
76,0,1,2,0
28,0,6,0,1
43,0,4,1,1
36,1,0,1,0
21,0,5,2,0
20,0,0,2,0
82,1,4,0,1
58,0,5,0,0
72,1,6,1,0
38,0,1,1,0
75,0,0,2,1
68,0,4,1,0
44,0,4,1,0
46,0,5,0,0
63,0,0,0,0
38,1,5,1,1
34,0,1,0,1
63,1,2,0,0
67,0,1,0,0
48,1,2,1,0
58,0,6,1,0
59,1,0,1,0
72,0,4,2,0
73,1,4,1,1
51,0,7,0,0
34,0,0,2,1
38,0,7,2,0
63,0,2,0,1
34,1,3,1,0
23,0,0,1,1
78,1,5,0,0
43,1,5,0,0
30,0,4,2,0
25,0,4,0,1
33,1,4,1,0
26,0,3,1,0
70,0,7,2,1
57,0,3,1,1
74,0,3,2,0
81,1,0,1,0
49,0,1,0,0
26,0,4,1,0
81,0,5,1,1
65,1,6,2,0
31,0,6,0,0
58,1,1,1,0
22,1,6,1,1
77,0,6,1,1
30,1,5,2,0
42,0,1,2,1
67,0,4,1,0
24,0,1,0,1
84,0,3,0,0
73,0,1,2,0
55,0,0,2,0
23,1,7,2,0
40,0,5,1,0
51,1,6,0,1
83,1,4,0,0
63,1,1,2,0
18,1,0,2,0
23,0,3,0,0
58,1,1,2,1
27,1,5,1,1
59,1,2,2,1
27,1,3,0,0
31,1,2,2,0
19,0,0,0,0
29,0,6,1,0
18,1,1,2,0
24,1,5,2,1
27,1,5,1,1
57,0,5,0,1
74,0,0,2,0
22,0,1,1,0
33,0,3,2,0
57,1,2,2,0
61,1,2,1,0
20,1,6,2,1
67,1,4,1,1
80,0,5,0,1
44,1,6,2,0
26,0,3,2,1
80,1,4,2,0
63,1,3,2,0
58,0,3,2,0
38,0,0,0,1
60,1,2,0,0
49,1,2,0,1
80,1,0,2,1
35,0,2,0,0
84,1,0,1,0
76,1,0,0,0
79,0,1,0,1
76,0,6,0,1
55,0,0,0,0
53,1,3,0,0
18,1,2,0,1
"""
import io
data = pd.read_csv(io.StringIO(csv_text))

# public columns and target
pub = ["age", "sex", "blood", "admission"]
target = "result"

# ---------------- Query interfaces ----------------

def execute_subsetsums_exact(predicates):
    """Exact answers: returns numpy array shape (k,)"""
    return data[target].values @ np.stack([pred(data) for pred in predicates], axis=1)

def execute_subsetsums_round(R, predicates):
    exact = execute_subsetsums_exact(predicates).astype(float)
    return (np.round(exact / R) * R).astype(float)

def execute_subsetsums_noise(sigma, predicates):
    exact = execute_subsetsums_exact(predicates).astype(float)
    noise = np.random.normal(0.0, float(sigma), size=exact.shape[0])
    return exact + noise

def execute_subsetsums_sample(t, predicates):
    n = len(data)
    t = max(1, min(int(t), n))
    idx = np.random.choice(n, size=t, replace=False)
    sub = data.iloc[idx].reset_index(drop=True)
    answers = sub[target].values @ np.stack([pred(sub) for pred in predicates], axis=1)
    return answers.astype(float) * (n / t)

# ---------- Random predicate factory ----------

def make_random_predicate():
    prime = 2003
    desc = np.random.randint(prime, size=len(pub))
    return lambda df: ((df[pub].values @ desc) % prime % 2).astype(bool)

# ---------- Reconstruction attack ----------

def reconstruction_attack(data_pub, predicates, answers):
    """Least-squares attack: solve for r, then threshold to {0,1}."""
    A = np.stack([pred(data_pub).astype(float) for pred in predicates], axis=1)  # (n, k)
    y = np.asarray(answers, dtype=float)
    # Solve A^T r = y  (least squares)
    r_hat, *_ = np.linalg.lstsq(A.T, y, rcond=None)
    r_hat = np.clip(r_hat, 0.0, 1.0)
    return (r_hat >= 0.5).astype(int)
