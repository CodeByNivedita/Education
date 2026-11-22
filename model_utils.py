"""Training code shared by app.py and train.py.

The app trains the model itself the first time it starts, so the model that makes
predictions always comes from the same Python setup that runs the app.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

TARGET = "Post_Semester_GPA"
# Measured AFTER the semester, so a student can't know them in advance.
# Using them as inputs would leak the outcome, so they are excluded.
OUTCOME_COLS = ["Skill_Retention_Score", "Burnout_Risk_Level"]
ID_COLS = ["Student_ID"]


def _pipeline(regressor, num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore",
                                                    sparse_output=False))]), cat_cols),
    ])
    return Pipeline([("preprocessor", pre), ("regressor", regressor)])


def train(csv_path="student.csv"):
    """Return (fitted model, meta dict). Falls back to a second model type if the
    first one fails a basic sanity check (accurate and predictions actually vary)."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    for c in df.select_dtypes(include="bool").columns:  # True/False -> text categories
        df[c] = df[c].astype(str)

    X = df.drop(columns=[TARGET] + [c for c in OUTCOME_COLS if c in df.columns])
    y = df[TARGET]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = [
        ("XGBoost", XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
        ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=42)),
    ]
    for name, reg in candidates:
        model = _pipeline(reg, num_cols, cat_cols)
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_test), dtype=float)
        ok = np.isfinite(pred).all() and r2_score(y_test, pred) > 0.3 and pred.std() > 0.05
        if ok:
            break

    imp = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=42)
    importance = dict(sorted(zip(X.columns, imp.importances_mean.tolist()),
                             key=lambda kv: kv[1], reverse=True))
    meta = {
        "model_name": name,
        "sanity_check_passed": bool(ok),
        "rows": int(len(df)),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "features": list(X.columns),
        "numeric": {
            c: {"min": float(X[c].min()), "max": float(X[c].max()),
                "median": float(X[c].median()),
                "is_int": bool(pd.api.types.is_integer_dtype(X[c]))}
            for c in num_cols
        },
        "categorical": {
            c: {"options": sorted(X[c].dropna().astype(str).unique().tolist()),
                "default": str(X[c].mode().iloc[0])}
            for c in cat_cols
        },
        "excluded_outcomes": [c for c in OUTCOME_COLS if c in df.columns],
        "metrics": {
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": float(r2_score(y_test, pred)),
        },
        "pred_min": float(pred.min()), "pred_mean": float(pred.mean()),
        "pred_max": float(pred.max()),
        "importance": importance,
    }
    return model, meta