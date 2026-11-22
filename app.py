from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from model_utils import train

st.set_page_config(page_title="AI & Student GPA", page_icon="🎓", layout="wide")


def label(col: str) -> str:
    return col.replace("_", " ")


@st.cache_resource(show_spinner="Training the model on student.csv (about 20 seconds, first start only)...")
def get_model():
    return train("student.csv")


@st.cache_data
def load_data():
    return pd.read_csv("student.csv")


if not Path("student.csv").exists():
    st.error("student.csv not found. Put the Kaggle CSV in the same folder as app.py and rerun.")
    st.stop()

model, meta = get_model()
data = load_data()
features = meta["features"]


def predict_raw(rows: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(rows[features]))


def predict(rows: pd.DataFrame) -> pd.Series:
    return predict_raw(rows).clip(meta["target_min"], meta["target_max"])


# ---------- Sidebar: one student profile shared by every tab ----------
st.sidebar.header("Student profile")
row = {}
for c in features:
    if c in meta["numeric"]:
        s = meta["numeric"][c]
        lo, hi, med = s["min"], s["max"], s["median"]
        if hi <= lo:
            row[c] = lo
        elif s["is_int"]:
            row[c] = st.sidebar.slider(label(c), int(lo), int(hi), int(round(med)))
        else:
            step = round((hi - lo) / 100, 2) or 0.01
            row[c] = st.sidebar.slider(label(c), float(lo), float(hi), float(med), step=step)
    else:
        opts = meta["categorical"][c]["options"]
        default = opts.index(meta["categorical"][c]["default"])
        row[c] = st.sidebar.selectbox(label(c), opts, index=default)

st.title("How does AI use relate to a student's GPA?")
st.caption("Trained on 50,000 student records. Predictions show patterns in the data, "
           "not proof that AI use causes a change in GPA.")

tab_predict, tab_whatif, tab_insights = st.tabs(["Predict GPA", "What-if explorer", "Data insights"])

# ---------- Tab 1: prediction ----------
with tab_predict:
    # "Typical student" = the sidebar defaults (median / most common value of every input)
    baseline = {}
    for c in features:
        if c in meta["numeric"]:
            s = meta["numeric"][c]
            baseline[c] = int(round(s["median"])) if s["is_int"] else s["median"]
        else:
            baseline[c] = meta["categorical"][c]["default"]

    raw = float(predict_raw(pd.DataFrame([row])).iloc[0])
    gpa = float(predict(pd.DataFrame([row])).iloc[0])
    base_gpa = float(predict(pd.DataFrame([baseline])).iloc[0])
    pre = row.get("Pre_Semester_GPA")
    t_lo, t_hi = meta["target_min"], meta["target_max"]

    left, right = st.columns(2)
    left.metric("Predicted post-semester GPA", f"{gpa:.3f}",
                delta=f"{gpa - base_gpa:+.3f} vs a typical student")
    if pre is not None:
        right.metric("Change from pre-semester GPA", f"{gpa - pre:+.3f}")

    if raw < t_lo or raw > t_hi:
        st.warning(f"The model's raw output was {raw:.3f}, outside the GPA range in the data "
                   f"({t_lo:.2f} to {t_hi:.2f}), so the app capped it. A healthy model stays inside "
                   "that range, so the model or the CSV needs checking.")
    with st.expander("Model check"):
        m = meta["metrics"]
        st.write(f"Raw model output for this profile: **{raw:.3f}**")
        st.write(f"GPA range seen in the data: **{t_lo:.2f} to {t_hi:.2f}**")
        st.write(f"Accuracy on held-out students: R² **{m['R2']:.3f}**, MAE **{m['MAE']:.3f}**")
        st.write(f"Model type: **{meta['model_name']}** | trained on **{meta['rows']:,}** rows | "
                 f"predictions on test students range **{meta['pred_min']:.2f} to {meta['pred_max']:.2f}**")
        if not meta["sanity_check_passed"]:
            st.error("The model failed its own sanity check (low accuracy or constant predictions).")
        st.write("Inputs the model uses: " + ", ".join(label(c) for c in features))

    st.subheader("Which inputs moved the prediction?")
    # Change ONE input at a time from the typical student to your value, and see the effect
    one_at_a_time = pd.DataFrame([baseline] * len(features))
    for i, c in enumerate(features):
        one_at_a_time.at[i, c] = row[c]
    effect = pd.Series(predict(one_at_a_time).values - base_gpa,
                       index=[label(c) for c in features], name="Effect on predicted GPA")
    st.bar_chart(effect, horizontal=True)
    st.caption("Each bar shows how much that one input moves the predicted GPA compared with a typical "
               "student. If a bar stays near 0 when you move its slider, the data shows little "
               "relationship between that input and GPA.")

# ---------- Tab 2: what-if ----------
with tab_whatif:
    default_idx = next((i for i, c in enumerate(features)
                        if "genai" in c.lower() and "hour" in c.lower()), 0)
    var = st.selectbox("Vary this input, keep the rest as in the sidebar",
                       features, index=default_idx, format_func=label)
    if var in meta["numeric"]:
        s = meta["numeric"][var]
        xs = [s["min"] + i * (s["max"] - s["min"]) / 40 for i in range(41)]
        sweep = pd.DataFrame([row] * len(xs))
        sweep[var] = xs
        plot = pd.DataFrame({"x": xs, "y": predict(sweep).values})
    else:
        opts = meta["categorical"][var]["options"]
        sweep = pd.DataFrame([row] * len(opts))
        sweep[var] = opts
        plot = pd.DataFrame({"x": opts, "y": predict(sweep).values})

    lo, hi = plot["y"].min(), plot["y"].max()
    pad = max((hi - lo) * 0.25, 0.005)
    # Zoom the y-axis to the prediction range so small differences are visible
    y = alt.Y("y:Q", title="Predicted GPA", scale=alt.Scale(domain=[lo - pad, hi + pad], zero=False))
    if var in meta["numeric"]:
        chart = alt.Chart(plot).mark_line(point=True).encode(x=alt.X("x:Q", title=label(var)), y=y)
    else:
        chart = alt.Chart(plot).mark_point(filled=True, size=160).encode(
            x=alt.X("x:N", title=label(var)), y=y)
    st.altair_chart(chart)
    st.write(f"Across this range the predicted GPA goes from **{lo:.3f}** to **{hi:.3f}** "
             f"(a spread of {hi - lo:.3f} GPA points).")
    st.caption("The y-axis is zoomed in, so small changes look bigger than they are. "
               "Check the spread above for the real size of the effect.")

# ---------- Tab 3: insights ----------
with tab_insights:
    st.subheader("How well the model predicts")
    m = meta["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{m['R2']:.3f}")
    c2.metric("MAE", f"{m['MAE']:.3f}")
    c3.metric("RMSE", f"{m['RMSE']:.3f}")
    if meta["excluded_outcomes"]:
        st.caption(", ".join(label(c) for c in meta["excluded_outcomes"]) +
                   " are not model inputs: they are measured after the semester.")

    st.subheader("What the model relies on most")
    st.bar_chart(pd.Series(meta["importance"], name="Importance (drop in R² when shuffled)"),
                 horizontal=True)

    st.subheader("Explore the data")
    if data is None:
        st.info("Add student.csv next to app.py to see charts of the raw data.")
    else:
        d = data.copy()
        outcomes = {}
        if {"Pre_Semester_GPA", "Post_Semester_GPA"} <= set(d.columns):
            d["GPA change"] = d["Post_Semester_GPA"] - d["Pre_Semester_GPA"]
            outcomes["GPA change"] = "GPA change"
        if "Skill_Retention_Score" in d.columns:
            outcomes["Skill retention score"] = "Skill_Retention_Score"
        skip = {"Student_ID", "Post_Semester_GPA", "GPA change"}
        by_opts = [c for c in d.columns if c not in skip and c not in outcomes.values()]
        a, b = st.columns(2)
        out_name = a.selectbox("Outcome (average)", list(outcomes))
        by = b.selectbox("Grouped by", by_opts, format_func=label)
        out_col = outcomes[out_name]
        if pd.api.types.is_numeric_dtype(d[by]) and d[by].nunique() > 12:
            g = d.groupby(pd.qcut(d[by], 10, duplicates="drop"), observed=True)[out_col].mean()
            series = pd.Series(g.values, index=[round(i.mid, 1) for i in g.index])
        else:
            series = d.groupby(by)[out_col].mean()
            series.index = series.index.astype(str)
        st.bar_chart(series, x_label=label(by), y_label=f"Average {out_name.lower()}")