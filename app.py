"""GlucoGuard — AI diabetes risk screening, glucose tracking and lifestyle coaching (Streamlit + scikit-learn)."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import auth
import model as M
import theme
from ui import badge, hero, kpis, result_card, sidebar_brand

APP = dict(
    name="GlucoGuard", short="Diabetes risk AI", icon="🩸", tagline="Know your diabetes risk — early.",
    description="Machine-learning risk screening with or without lab reports, the Indian Diabetes Risk Score, personal explanations, what-if lifestyle coaching and a glucose diary.",
    features=["4 models incl. a neural net · AUC ≈ 0.96", "No-lab screening + Indian Diabetes Risk Score (IDRS)", "Explains every prediction & simulates lifestyle changes", "Glucose diary with estimated HbA1c"],
    card1="🩸 Risk 18% · Moderate · IDRS 50", card2="🚶 +150 min walking → risk −6%",
    demo_name="Bharath Kumar", demo_email="demo@glucoguard.app", demo_password="Demo@1234",
    roles=["Individual", "Doctor", "Health worker", "Researcher"],
)
st.set_page_config(page_title="GlucoGuard — Diabetes Risk AI", page_icon="🩸", layout="wide")
theme.inject("Ocean")


@st.cache_data(show_spinner="Loading screening cohort…")
def load():
    return M.generate()


@st.cache_resource(show_spinner="Training risk models…")
def models():
    d = load()
    return M.train(d), M.train_nolab(d)


if "user" not in st.session_state:
    auth.auth_page(APP)
    st.stop()

user = st.session_state.user
sidebar_brand(APP, user)
PAGES = ["🧪 Risk checker", "📓 Glucose diary", "🌍 Population insights", "📂 Batch screening", "🤖 Model lab", "👤 Account"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
mname = st.sidebar.selectbox("🧠 Model", list(M.MODELS))
theme.theme_controls()
if st.sidebar.button("Log out", width="stretch"):
    auth.logout()
st.sidebar.caption("⚕️ Screening tool only — not a diagnosis. Consult a doctor.")

R, (NOLAB, NOLAB_AUC) = models()
D = load()
PIPE = R["models"][mname]["pipe"]
LR = R["models"]["Logistic Regression"]["pipe"]

# ------------------------------------------------------------------ Risk checker
if page == PAGES[0]:
    hero("Diabetes risk checker", "Answer a few questions — get your risk, the reasons behind it and a plan to lower it", "🧪")
    has_lab = st.toggle("I have recent blood-test results (fasting glucose / HbA1c)", value=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### 👤 About you")
        age = st.slider("Age", 18, 85, 42)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        fam = st.selectbox("Family history of diabetes", ["None", "One parent", "Both parents"], index=1)
        smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])
    with c2:
        st.markdown("##### 📏 Body")
        h = st.slider("Height (cm)", 140, 200, 170)
        w = st.slider("Weight (kg)", 35, 150, 78)
        bmi = round(w / (h / 100) ** 2, 1)
        waist = st.slider("Waist (cm)", 55, 150, 94)
        sbp = st.slider("Systolic BP (mmHg)", 90, 200, 128)
        st.caption(f"BMI = **{bmi}** ({'normal' if bmi < 23 else 'overweight (Asian cut-off ≥ 23)' if bmi < 27.5 else 'obese (Asian cut-off ≥ 27.5)'})")
    with c3:
        st.markdown("##### 🏃 Lifestyle")
        act = st.slider("Exercise (min / week)", 0, 600, 90, 10)
        diet = st.selectbox("Typical diet", ["Balanced", "High-carb", "Junk-heavy", "Low-carb"], index=1)
        sleep = st.slider("Sleep (hours / night)", 3.5, 10.0, 6.5, 0.5)
        fpg = st.slider("Fasting glucose (mg/dL)", 65, 300, 104, disabled=not has_lab)
        a1c = st.slider("HbA1c (%)", 4.0, 13.0, 5.8, 0.1, disabled=not has_lab)
        chol = st.slider("Total cholesterol (mg/dL)", 110, 330, 205, disabled=not has_lab)
    row = dict(age=age, gender=gender, bmi=bmi, waist_cm=waist, fasting_glucose=fpg, hba1c=a1c, systolic_bp=sbp, cholesterol=chol, sleep_hours=sleep, activity_min=act, family_history=fam, diet=diet, smoking=smoking)
    p = float((PIPE if has_lab else NOLAB).predict_proba(pd.DataFrame([row]))[0, 1])
    lvl, col = M.band(p)
    score = M.idrs(age, waist, gender, act, fam)
    st.write("")
    a, b, c = st.columns([1.1, 1, 1])
    with a:
        result_card("Predicted diabetes risk", f"{p * 100:.0f}% · {lvl}", f"{'Lab-based model: ' + mname if has_lab else f'No-lab screening model (AUC {NOLAB_AUC:.2f})'} — compare with people your age: {D[(D.age - age).abs() <= 5].diabetes.mean() * 100:.0f}% prevalence", col, "#0c4a6e", "🩸")
    with b:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=score, title={"text": "Indian Diabetes Risk Score"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#ef4444" if score >= 60 else "#f59e0b" if score >= 30 else "#10b981"},
                                     "steps": [{"range": [0, 30], "color": "#10b98122"}, {"range": [30, 60], "color": "#f59e0b22"}, {"range": [60, 100], "color": "#ef444422"}]}))
        fig.update_layout(height=250)
        st.plotly_chart(fig, width="stretch")
        st.caption("IDRS (MDRF): < 30 low · 30–50 moderate · ≥ 60 high")
    with c:
        if has_lab:
            s, sc = M.stage_of(fpg, a1c)
            st.markdown(f'<div class="card" style="border-left:6px solid {sc}"><small>Clinical glucose range (ADA)</small><h3 style="margin:4px 0;color:{sc}">{s}</h3><small>Fasting {fpg} mg/dL · HbA1c {a1c}%</small><br><small>Normal &lt; 100 / &lt; 5.7 · Prediabetes 100–125 / 5.7–6.4 · Diabetes ≥ 126 / ≥ 6.5</small></div>', unsafe_allow_html=True)
        else:
            st.info("Add blood-test values for a more accurate, lab-based estimate.")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        if has_lab:
            ex = M.contributions(LR, row).head(10)[::-1]
            fig = go.Figure(go.Bar(x=ex.values, y=ex.index, orientation="h", marker_color=["#ef4444" if v > 0 else "#10b981" for v in ex.values]))
            fig.update_layout(title="What drives your risk (red raises, green lowers)", height=380, xaxis_title="log-odds contribution")
            st.plotly_chart(fig, width="stretch")
        st.markdown("##### 🔮 What if you changed your lifestyle?")
        k1, k2, k3 = st.columns(3)
        lose = k1.slider("Lose weight (kg)", 0, 20, 5)
        more = k2.slider("Extra exercise (min/wk)", 0, 300, 90, 10)
        newdiet = k3.selectbox("Switch diet to", ["(no change)", "Balanced", "Low-carb"])
        r2 = dict(row, bmi=round((w - lose) / (h / 100) ** 2, 1), waist_cm=waist - lose * 0.9, activity_min=act + more, diet=diet if newdiet == "(no change)" else newdiet,
                  fasting_glucose=fpg - lose * 0.8, hba1c=round(a1c - lose * 0.03, 1))
        p2 = float((PIPE if has_lab else NOLAB).predict_proba(pd.DataFrame([r2]))[0, 1])
        fig = go.Figure(go.Bar(x=["Today", "With changes"], y=[p * 100, p2 * 100], marker_color=[col, M.band(p2)[1]], text=[f"{p * 100:.0f}%", f"{p2 * 100:.0f}%"], textposition="outside"))
        fig.update_layout(height=280, yaxis_range=[0, max(p, p2) * 130], yaxis_title="risk %", title=f"Risk change: {(p2 - p) * 100:+.1f} points")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("##### 🧭 Your personal plan")
        for i, t in M.plan(row):
            st.markdown(f'<div class="card" style="margin-bottom:8px">{i} {t}</div>', unsafe_allow_html=True)
        st.session_state.setdefault("checks", [])
        if st.button("💾 Save this check to my history", type="primary"):
            st.session_state.checks.append({"date": str(date.today()), "risk %": round(p * 100, 1), "IDRS": score, "BMI": bmi, "waist": waist, "exercise": act})
            st.toast("Saved ✅")
        if st.session_state.checks:
            st.dataframe(pd.DataFrame(st.session_state.checks), hide_index=True, width="stretch")

# ------------------------------------------------------------------ Glucose diary
elif page == PAGES[1]:
    hero("Glucose diary", "Log readings, see trends against target ranges and estimate your HbA1c", "📓")
    if "diary" not in st.session_state:
        rng = np.random.default_rng(1)
        rows = []
        for i in range(30, 0, -1):
            d = date.today() - timedelta(days=i)
            rows.append({"date": d, "type": "Fasting", "mg/dL": int(rng.normal(112 - i * 0.25, 9))})
            rows.append({"date": d, "type": "Post-meal (2h)", "mg/dL": int(rng.normal(158 - i * 0.4, 16))})
        st.session_state.diary = rows
    with st.form("log", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 0.6])
        d = c1.date_input("Date", date.today())
        t = c2.selectbox("Reading", ["Fasting", "Post-meal (2h)", "Random", "Bedtime"])
        v = c3.number_input("Glucose (mg/dL)", 40, 500, 110)
        if c4.form_submit_button("➕ Add", type="primary"):
            st.session_state.diary.append({"date": d, "type": t, "mg/dL": int(v)})
            st.toast("Reading logged 🩸")
    L = pd.DataFrame(st.session_state.diary).sort_values("date")
    avg = L["mg/dL"].mean()
    fast = L[L.type == "Fasting"]["mg/dL"]
    inrange = ((L.type == "Fasting") & L["mg/dL"].between(70, 100) | (L.type != "Fasting") & L["mg/dL"].between(70, 140)).mean() * 100
    kpis([("🩸", "Average glucose", f"{avg:.0f} mg/dL", f"{len(L)} readings"), ("🧮", "Estimated HbA1c", f"{M.eag_to_a1c(avg)}%", "ADAG formula"), ("🎯", "Time in target", f"{inrange:.0f}%", "of readings"), ("🌅", "Avg fasting", f"{fast.mean():.0f}", "target < 100")])
    fig = px.scatter(L, x="date", y="mg/dL", color="type", title="Readings over time")
    fig.add_hrect(y0=70, y1=100, fillcolor="#10b981", opacity=0.1, line_width=0, annotation_text="fasting target")
    fig.add_hrect(y0=100, y1=140, fillcolor="#f59e0b", opacity=0.07, line_width=0, annotation_text="post-meal target")
    for typ in L.type.unique():
        s = L[L.type == typ].groupby("date")["mg/dL"].mean().rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=f"{typ} 7-day avg", line=dict(width=3)))
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    fig = px.histogram(L, x="mg/dL", color="type", nbins=30, barmode="overlay", title="Distribution")
    c1.plotly_chart(fig, width="stretch")
    c2.dataframe(L.sort_values("date", ascending=False).head(20), hide_index=True, width="stretch")
    st.download_button("⬇️ Export diary (CSV)", L.to_csv(index=False), "glucoguard_diary.csv", "text/csv")

# ------------------------------------------------------------------ Population
elif page == PAGES[2]:
    hero("Population insights", f"{len(D):,} screened adults · prevalence {D.diabetes.mean() * 100:.1f}%", "🌍")
    D2 = D.assign(age_group=pd.cut(D.age, [17, 30, 40, 50, 60, 85], labels=["18–30", "31–40", "41–50", "51–60", "60+"]), bmi_group=pd.cut(D.bmi, [0, 23, 27.5, 60], labels=["Normal", "Overweight", "Obese"]))
    kpis([("👥", "Screened", f"{len(D):,}", ""), ("🩸", "Diabetes", f"{D.diabetes.mean() * 100:.1f}%", ""), ("⚠️", "Prediabetes", f"{(D.stage == 'Prediabetes').mean() * 100:.1f}%", "often undiagnosed"), ("⚖️", "Overweight/obese", f"{(D.bmi >= 23).mean() * 100:.0f}%", "Asian cut-offs")])
    c1, c2 = st.columns(2)
    g = D2.groupby(["age_group", "stage"], observed=True).size().reset_index(name="n")
    fig = px.bar(g, x="age_group", y="n", color="stage", color_discrete_map={"Normal": "#10b981", "Prediabetes": "#f59e0b", "Diabetes": "#ef4444"}, title="Glycaemic status by age")
    fig.update_layout(barnorm="percent", yaxis_title="%")
    c1.plotly_chart(fig, width="stretch")
    hm = D2.pivot_table(index="bmi_group", columns="family_history", values="diabetes", aggfunc="mean", observed=True) * 100
    fig = px.imshow(hm, text_auto=".0f", color_continuous_scale="Reds", title="Diabetes % by BMI × family history")
    c2.plotly_chart(fig, width="stretch")
    fig = px.scatter(D.sample(1500, random_state=2), x="bmi", y="fasting_glucose", color="stage", opacity=0.55, color_discrete_map={"Normal": "#10b981", "Prediabetes": "#f59e0b", "Diabetes": "#ef4444"}, title="BMI vs fasting glucose")
    fig.add_hline(y=126, line_dash="dash")
    fig.add_hline(y=100, line_dash="dot")
    st.plotly_chart(fig, width="stretch")
    act = D.assign(act=pd.cut(D.activity_min, [-1, 60, 150, 300, 700], labels=["< 60", "60–150", "150–300", "300+"])).groupby(["act", "diet"], observed=True).diabetes.mean().reset_index()
    fig = px.bar(act, x="act", y="diabetes", color="diet", barmode="group", title="Diabetes rate by weekly exercise and diet")
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title="exercise (min/week)")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Batch
elif page == PAGES[3]:
    hero("Batch screening", "Upload a camp / clinic list and flag high-risk people for follow-up", "📂")
    st.download_button("⬇️ CSV template", D[M.FEATURES].head(5).to_csv(index=False), "glucoguard_template.csv", "text/csv")
    up = st.file_uploader("Upload screening CSV", type=["csv"])
    df = pd.read_csv(up) if up else D.sample(60, random_state=5)[M.FEATURES].reset_index(drop=True)
    if not up:
        st.caption("Showing a 60-person sample camp.")
    miss = [c for c in M.FEATURES if c not in df.columns]
    if miss:
        st.error(f"Missing columns: {', '.join(miss)}")
    else:
        out = df.copy()
        out["risk"] = PIPE.predict_proba(df[M.FEATURES])[:, 1].round(3)
        out["band"] = out.risk.map(lambda v: M.band(v)[0])
        out["IDRS"] = [M.idrs(r.age, r.waist_cm, r.gender, r.activity_min, r.family_history) for r in df.itertuples()]
        out = out.sort_values("risk", ascending=False)
        kpis([("👥", "People", f"{len(out)}", ""), ("🔴", "High risk", f"{(out.band == 'High').sum()}", "refer for HbA1c"), ("🟠", "Moderate", f"{(out.band == 'Moderate').sum()}", "lifestyle counselling"), ("🟢", "Low", f"{(out.band == 'Low').sum()}", "")])
        st.dataframe(out, hide_index=True, width="stretch", column_config={"risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
        st.download_button("⬇️ Download results", out.to_csv(index=False), "glucoguard_screening.csv", "text/csv", type="primary")

# ------------------------------------------------------------------ Model lab
elif page == PAGES[4]:
    hero("Model lab", "1,000 held-out people · stratified split · probability calibration matters for medicine", "🤖")
    tab = pd.DataFrame({n: {k: v[k] for k in ["AUC", "Avg precision", "Brier", "Accuracy", "F1"]} for n, v in R["models"].items()}).T
    st.dataframe(tab.style.format("{:.3f}").highlight_max(subset=["AUC", "Avg precision", "Accuracy", "F1"], color="#10b98144").highlight_min(subset=["Brier"], color="#10b98144"), width="stretch")
    st.caption(f"No-lab lifestyle model AUC: {NOLAB_AUC:.3f} — useful for community screening before any blood test.")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = go.Figure()
        for n, v in R["models"].items():
            fig.add_trace(go.Scatter(x=v["fpr"], y=v["tpr"], name=f"{n} ({v['AUC']:.3f})"))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot"))
        fig.update_layout(title="ROC curves", height=360, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        for n, v in R["models"].items():
            fig.add_trace(go.Scatter(x=v["cal"][0], y=v["cal"][1], mode="lines+markers", name=n))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot"))
        fig.update_layout(title="Calibration (closer to diagonal = trustworthy %)", height=360, xaxis_title="predicted", yaxis_title="observed", legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch")
    with c3:
        thr = st.slider("Decision threshold", 0.05, 0.95, 0.3, 0.05)
        pr = R["models"][mname]["proba"]
        y = R["yte"]
        tp, fp = int(((pr >= thr) & (y == 1)).sum()), int(((pr >= thr) & (y == 0)).sum())
        fn, tn = int(((pr < thr) & (y == 1)).sum()), int(((pr < thr) & (y == 0)).sum())
        fig = px.imshow([[tn, fp], [fn, tp]], x=["Pred. no", "Pred. yes"], y=["Actual no", "Actual yes"], text_auto=True, color_continuous_scale="Blues", title=f"Confusion @ {thr:.2f}")
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Sensitivity {tp / max(tp + fn, 1):.2f} · Specificity {tn / max(tn + fp, 1):.2f} — screening favours high sensitivity.")
    rf = R["models"]["Random Forest"]["pipe"]
    imp = pd.Series(rf.named_steps["m"].feature_importances_, index=rf.named_steps["pre"].get_feature_names_out()).sort_values().tail(12)
    imp.index = [i.split("__", 1)[1] for i in imp.index]
    fig = px.bar(imp, orientation="h", title="Feature importance (Random Forest)")
    fig.update_layout(height=380, showlegend=False, yaxis_title="")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Account
else:
    hero(f"Hi, {user['name'].split()[0]} 👋", f"{user['email']} · {user['role']} · member since {user['created'][:10]}", "👤")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🔒 Change password")
        with st.form("pw"):
            old = st.text_input("Current password", type="password")
            new = st.text_input("New password", type="password")
            if st.form_submit_button("Update password", type="primary"):
                try:
                    auth.change_password(user["email"], old, new)
                    st.success("Password updated")
                except ValueError as e:
                    st.error(str(e))
    with c2:
        st.markdown("##### 🗂️ Data")
        st.markdown(f'<div class="card">{len(D):,} simulated screening records with 13 risk factors. Registered users: <b>{auth.user_count()}</b>.</div>', unsafe_allow_html=True)
        st.download_button("⬇️ Download cohort (CSV)", D.to_csv(index=False), "glucoguard_cohort.csv", "text/csv")
