"""Helper functions: data loading, KPIs, SQL engine and delivery-time model."""
import math
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "orders.csv"
SQL_PATH = ROOT / "sql" / "queries.sql"

# Same constants used in generate_data.py to build the "promised time"
CUISINE_PREP = {"North Indian": 17, "Biryani": 19, "Fast Food": 11, "Chinese": 14,
                "South Indian": 12, "Pizza": 18, "Desserts": 8, "Healthy": 13}

CAT_FEATURES = ["traffic", "weather", "city", "cuisine"]
NUM_FEATURES = ["distance_km", "prep_time_min", "order_hour", "n_items"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
    df["month"] = df["order_date"].dt.to_period("M").astype(str)
    df["weekday"] = df["order_date"].dt.day_name()
    df["is_weekend"] = df["order_date"].dt.dayofweek >= 5
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    delivered = df[df["cancelled"] == 0]
    return {
        "orders": len(df),
        "revenue": float(delivered["order_value"].sum()),
        "aov": float(delivered["order_value"].mean()) if len(delivered) else 0.0,
        "avg_delivery": float(df["delivery_time_min"].mean()) if len(df) else 0.0,
        "late_pct": float(100 * df["is_late"].mean()) if len(df) else 0.0,
        "cancel_pct": float(100 * df["cancelled"].mean()) if len(df) else 0.0,
        "avg_rating": float(delivered["customer_rating"].mean()) if len(delivered) else 0.0,
    }


def promised_time(distance_km: float, cuisine: str) -> int:
    raw = 12 + distance_km * 3.6 + CUISINE_PREP[cuisine]
    return int(math.ceil(raw / 5) * 5 + 5)


# ---------------------------------------------------------------- SQL explorer
def load_queries() -> dict:
    """Parse sql/queries.sql into {name: {'desc': ..., 'sql': ...}}."""
    text = SQL_PATH.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^-- name:\s*", text)[1:]
    out = {}
    for b in blocks:
        name, _, rest = b.partition("\n")
        desc = ""
        m = re.match(r"-- desc:\s*(.*)\n", rest)
        if m:
            desc = m.group(1).strip()
            rest = rest[m.end():]
        out[name.strip()] = {"desc": desc, "sql": rest.strip()}
    return out


def run_sql(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Run a query against an in-memory SQLite copy of the (filtered) orders."""
    con = sqlite3.connect(":memory:")
    try:
        tmp = df.drop(columns=["month", "weekday", "is_weekend"], errors="ignore").copy()
        tmp["order_date"] = tmp["order_date"].dt.strftime("%Y-%m-%d")
        tmp.to_sql("orders", con, index=False)
        return pd.read_sql_query(query, con)
    finally:
        con.close()


# ---------------------------------------------------------------------- model
def train_model(df: pd.DataFrame):
    """Train a gradient-boosting model to predict delivery time (minutes)."""
    X = df[CAT_FEATURES + NUM_FEATURES]
    y = df["delivery_time_min"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)],
        remainder="passthrough",
    )
    model = Pipeline([("prep", pre),
                      ("gbm", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                                            random_state=42))])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    baseline = np.full(len(y_te), y_tr.mean())
    metrics = {
        "mae": mean_absolute_error(y_te, pred),
        "r2": r2_score(y_te, pred),
        "baseline_mae": mean_absolute_error(y_te, baseline),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }
    return model, metrics


def predict_one(model, **row) -> float:
    return float(model.predict(pd.DataFrame([row]))[0])
