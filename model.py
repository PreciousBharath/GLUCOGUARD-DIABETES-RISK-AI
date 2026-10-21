"""GlucoGuard engine: synthetic screening cohort, diabetes classifiers, calibration, IDRS score,
local explanations, lifestyle what-ifs and glucose-log utilities."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["age", "bmi", "waist_cm", "fasting_glucose", "hba1c", "systolic_bp", "cholesterol", "sleep_hours", "activity_min"]
CAT = ["gender", "family_history", "diet", "smoking"]
FEATURES = NUM + CAT
NICE = {"age": "Age", "bmi": "BMI", "waist_cm": "Waist (cm)", "fasting_glucose": "Fasting glucose (mg/dL)", "hba1c": "HbA1c (%)", "systolic_bp": "Systolic BP (mmHg)",
        "cholesterol": "Total cholesterol (mg/dL)", "sleep_hours": "Sleep (h/night)", "activity_min": "Exercise (min/week)", "gender": "Gender", "family_history": "Family history",
        "diet": "Diet", "smoking": "Smoking"}


def generate(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(44, 13, n), 18, 85).round()
    gender = rng.choice(["Male", "Female"], n)
    fam = rng.choice(["None", "One parent", "Both parents"], n, p=[0.55, 0.33, 0.12])
    diet = rng.choice(["Balanced", "High-carb", "Junk-heavy", "Low-carb"], n, p=[0.35, 0.35, 0.18, 0.12])
    smoke = rng.choice(["Never", "Former", "Current"], n, p=[0.68, 0.14, 0.18])
    activity = np.clip(rng.gamma(2, 70, n), 0, 600).round()
    bmi = np.clip(rng.normal(25.5, 4.5, n) + 0.04 * (age - 44) - 0.006 * (activity - 140) + 1.2 * (diet == "Junk-heavy"), 15, 48).round(1)
    waist = np.clip(bmi * 3.4 + np.where(gender == "Male", 8, 0) + rng.normal(0, 5, n), 55, 150).round()
    sleep = np.clip(rng.normal(6.9, 1.0, n), 3.5, 10).round(1)
    latent = (-7.4 + 0.045 * age + 0.13 * bmi + 0.02 * (waist - 85) + 0.75 * (fam == "One parent") + 1.5 * (fam == "Both parents") + 0.45 * (diet == "High-carb")
              + 0.8 * (diet == "Junk-heavy") - 0.35 * (diet == "Low-carb") - 0.004 * activity + 0.3 * (smoke == "Current") + 0.25 * (np.abs(sleep - 7.2) > 1.5) + rng.normal(0, 0.9, n))
    p = 1 / (1 + np.exp(-latent))
    diabetic = (rng.random(n) < p).astype(int)
    fpg = np.clip(np.where(diabetic, rng.normal(138, 34, n), 88 + 22 * p + rng.normal(0, 11, n)), 65, 350).round()
    a1c = np.clip(np.where(diabetic, 6.3 + (fpg - 126) / 38 + rng.normal(0, 0.45, n), 4.9 + 0.9 * p + rng.normal(0, 0.28, n)), 4.0, 13).round(1)
    sbp = np.clip(rng.normal(118, 13, n) + 0.45 * (age - 44) + 0.9 * (bmi - 25) + 6 * diabetic, 90, 200).round()
    chol = np.clip(rng.normal(185, 32, n) + 0.6 * (bmi - 25) * 2 + 12 * diabetic, 110, 330).round()
    df = pd.DataFrame({"age": age, "gender": gender, "bmi": bmi, "waist_cm": waist, "fasting_glucose": fpg, "hba1c": a1c, "systolic_bp": sbp, "cholesterol": chol,
                       "sleep_hours": sleep, "activity_min": activity, "family_history": fam, "diet": diet, "smoking": smoke, "diabetes": diabetic})
    df["stage"] = np.select([df.diabetes == 1, (df.hba1c >= 5.7) | (df.fasting_glucose >= 100)], ["Diabetes", "Prediabetes"], "Normal")
    return df


def _pre():
    return ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), CAT)])


MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, C=0.6),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=4, n_jobs=-1, random_state=0),
    "Gradient Boosting": lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, random_state=0),
    "Neural Net (MLP)": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=400, early_stopping=True, random_state=0),
}


def train(df):
    X, y = df[FEATURES], df["diabetes"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)
    out = {"yte": yte.values, "Xte": Xte, "models": {}}
    for n, f in MODELS.items():
        p = Pipeline([("pre", _pre()), ("m", f())]).fit(Xtr, ytr)
        pr = p.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, pr)
        prec, rec, _ = precision_recall_curve(yte, pr)
        cal_y, cal_x = calibration_curve(yte, pr, n_bins=10)
        out["models"][n] = {"pipe": p, "proba": pr, "AUC": roc_auc_score(yte, pr), "Avg precision": average_precision_score(yte, pr), "Brier": brier_score_loss(yte, pr),
                            "Accuracy": accuracy_score(yte, pr > 0.5), "F1": f1_score(yte, pr > 0.5), "fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec, "cal": (cal_x, cal_y)}
    return out


LIFESTYLE = ["age", "bmi", "waist_cm", "systolic_bp", "sleep_hours", "activity_min"] + CAT


def train_nolab(df):
    """Screening model that needs no blood test (for people without lab reports)."""
    pre = ColumnTransformer([("num", StandardScaler(), [c for c in LIFESTYLE if c in NUM]), ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), CAT)])
    X, y = df[LIFESTYLE], df["diabetes"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)
    p = Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=2000, C=0.6))]).fit(Xtr, ytr)
    return p, roc_auc_score(yte, p.predict_proba(Xte)[:, 1])


def idrs(age, waist, gender, activity_min, family):
    """Indian Diabetes Risk Score (Mohan et al., MDRF)."""
    s = 0 if age < 35 else 20 if age < 50 else 30
    lim1, lim2 = (90, 100) if gender == "Male" else (80, 90)
    s += 0 if waist < lim1 else 10 if waist < lim2 else 20
    s += 0 if activity_min >= 150 else 20 if activity_min >= 60 else 30
    s += {"None": 0, "One parent": 10, "Both parents": 20}[family]
    return s


def contributions(pipe, row):
    pre, m = pipe.named_steps["pre"], pipe.named_steps["m"]
    x = pre.transform(pd.DataFrame([row])[FEATURES])[0]
    names = pre.get_feature_names_out()
    c = pd.Series(x * m.coef_[0], index=names)
    agg = {}
    for k, v in c.items():
        base = k.split("__", 1)[1]
        key = next((cc for cc in CAT if base.startswith(cc + "_")), base)
        label = NICE[key] + (f": {row[key]}" if key in CAT else "")
        agg[label] = agg.get(label, 0) + v
    s = pd.Series(agg)
    return s[s.abs() > 0.02].sort_values(key=abs, ascending=False)


def band(p):
    return ("Low", "#10b981") if p < 0.15 else ("Moderate", "#f59e0b") if p < 0.4 else ("High", "#ef4444")


def stage_of(fpg, a1c):
    if a1c >= 6.5 or fpg >= 126:
        return "Diabetes range", "#ef4444"
    if a1c >= 5.7 or fpg >= 100:
        return "Prediabetes range", "#f59e0b"
    return "Normal range", "#10b981"


def plan(row):
    tips = []
    if row["bmi"] >= 25:
        tips.append(("⚖️", f"Lose 5–7 % of body weight — even 4–5 kg cuts diabetes risk by up to 58 % (DPP trial)."))
    if row["activity_min"] < 150:
        tips.append(("🚶", f"Build up to 150 min/week of brisk walking — you’re at {int(row['activity_min'])} min."))
    if row["diet"] in ("High-carb", "Junk-heavy"):
        tips.append(("🥗", "Swap refined rice/maida for millets, dal and vegetables; half-plate veggies at every meal."))
    if row["smoking"] == "Current":
        tips.append(("🚭", "Quit smoking — it raises insulin resistance and heart risk."))
    if row["sleep_hours"] < 6 or row["sleep_hours"] > 9:
        tips.append(("😴", "Aim for 7–8 hours of sleep; poor sleep worsens glucose control."))
    if row["fasting_glucose"] >= 100 or row["hba1c"] >= 5.7:
        tips.append(("🩸", "Repeat fasting glucose & HbA1c in 3 months and consult a physician."))
    if row["systolic_bp"] >= 130:
        tips.append(("❤️", "Blood pressure is elevated — cut salt and get it checked."))
    return tips or [("🌟", "Great habits! Keep up activity and get an annual check-up after 35.")]


def eag_to_a1c(avg_glucose):
    return round((avg_glucose + 46.7) / 28.7, 1)
