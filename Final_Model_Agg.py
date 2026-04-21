# =========================================================
# Walmart Department Sales Forecasting - Stable Full Version
# Models:
#   1. OLS
#   2. Random Forest
#   3. XGBoost
#   4. DeepAR (optional; if unavailable, script continues)
#
# Output:
#   - department_level_predictions_all_models.csv
#   - total_sales_predictions_all_models.csv
#   - actual_vs_models_total_sales.png
# =========================================================

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
# This forces torch.load to always use weights_only=False
_original_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_load
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# -----------------------------
# Optional XGBoost
# -----------------------------
HAS_XGB = True
try:
    import xgboost as xgb
except Exception as e:
    HAS_XGB = False
    print("\n[XGBoost unavailable]")
    print("Reason:", str(e))

# -----------------------------
# Optional DeepAR
# -----------------------------
HAS_DEEPAR = True
try:
    from gluonts.dataset.common import ListDataset
    from gluonts.torch.model.deepar import DeepAREstimator
except Exception as e:
    HAS_DEEPAR = False
    print("\n[DeepAR unavailable]")
    print("Reason:", str(e))

# =========================================================
# 0. SETTINGS
# =========================================================
DATA_DIR = "/Users/conniezhang/Desktop/ECON491/data/"

OUT_DEPT = "department_level_predictions_all_models.csv"
OUT_TOTAL = "total_sales_predictions_all_models.csv"
OUT_PLOT = "actual_vs_models_total_sales.png"

np.random.seed(123)

# =========================================================
# 1. HELPERS
# =========================================================
def print_step(msg):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def ensure_datetime(df, col="date"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def safe_sum_rmse(df, actual_col, pred_col):
    temp = df[[actual_col, pred_col]].dropna()
    if len(temp) == 0:
        return np.nan
    return rmse(temp[actual_col], temp[pred_col])

# =========================================================
# 2. LOAD DATA
# =========================================================
print_step("Loading Walmart data...")

os.chdir(DATA_DIR)

calendar = pd.read_csv("Calendar (1).csv")
sales_train = pd.read_csv("Sales Train Validation.csv")
sales_test  = pd.read_csv("Sales Test Validation.csv")
sell_prices = pd.read_csv("sell_prices.csv")

calendar["date"] = pd.to_datetime(calendar["date"])
calendar["d"] = [f"d_{i}" for i in range(1, len(calendar) + 1)]

calendar_use = calendar[
    [
        "d", "date", "wm_yr_wk", "weekday", "month", "year", "wday",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2"
    ]
].copy()

# =========================================================
# 3. BUILD DEPARTMENT-WEEK PRICE
# =========================================================
print_step("Building department weekly average price...")

item_dept_map = sales_train[["store_id", "item_id", "dept_id"]].drop_duplicates()

price_with_dept = sell_prices.merge(
    item_dept_map,
    on=["store_id", "item_id"],
    how="inner"
)

dept_week_price = (
    price_with_dept
    .groupby(["dept_id", "wm_yr_wk"], as_index=False)["sell_price"]
    .mean()
    .rename(columns={"sell_price": "avg_price"})
)

# =========================================================
# 4. BUILD DEPARTMENT-DAY PANEL
# =========================================================
def build_dept_day_panel(sales_wide, calendar_df, dept_week_price_df):
    id_vars = ["item_id", "dept_id", "store_id"]
    d_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    long_df = sales_wide.melt(
        id_vars=id_vars,
        value_vars=d_cols,
        var_name="d",
        value_name="sales"
    )

    long_df = long_df.merge(calendar_df, on="d", how="left")

    # fill event columns before groupby
    event_cols = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    for c in event_cols:
        long_df[c] = long_df[c].fillna("None")

    dept_day = (
        long_df
        .groupby(
            [
                "date", "dept_id", "wm_yr_wk", "weekday", "month", "year", "wday",
                "event_name_1", "event_type_1", "event_name_2", "event_type_2"
            ],
            as_index=False,
            dropna=False
        )["sales"]
        .sum()
        .rename(columns={"sales": "total_sales"})
    )

    dept_day = dept_day.merge(
        dept_week_price_df,
        on=["dept_id", "wm_yr_wk"],
        how="left"
    )

    dept_day["is_event"] = np.where(
        (
            (dept_day["event_name_1"] == "None") &
            (dept_day["event_type_1"] == "None") &
            (dept_day["event_name_2"] == "None") &
            (dept_day["event_type_2"] == "None")
        ),
        0, 1
    )

    dept_day = dept_day.sort_values(["dept_id", "date"]).reset_index(drop=True)
    return dept_day

print_step("Building department-day train/test panels...")

train_panel = build_dept_day_panel(sales_train, calendar_use, dept_week_price)
test_panel  = build_dept_day_panel(sales_test,  calendar_use, dept_week_price)

# =========================================================
# 5. FILL MISSING PRICE
# =========================================================
print_step("Filling missing prices...")

dept_price_mean = (
    train_panel.groupby("dept_id", as_index=False)["avg_price"]
    .mean()
    .rename(columns={"avg_price": "dept_price_mean"})
)

overall_price_mean = train_panel["avg_price"].mean()

train_panel = train_panel.merge(dept_price_mean, on="dept_id", how="left")
test_panel  = test_panel.merge(dept_price_mean, on="dept_id", how="left")

for df in [train_panel, test_panel]:
    df["avg_price"] = df["avg_price"].fillna(df["dept_price_mean"])
    df["avg_price"] = df["avg_price"].fillna(overall_price_mean)
    df.drop(columns=["dept_price_mean"], inplace=True)

# =========================================================
# 6. BUILD FEATURES USING TRAIN+TEST TO GET VALID LAGS
#    (for validation comparison)
# =========================================================
print_step("Creating features...")

train_panel["dataset"] = "train"
test_panel["dataset"] = "test"

all_data = pd.concat([train_panel, test_panel], ignore_index=True)
all_data = all_data.sort_values(["dept_id", "date"]).reset_index(drop=True)

all_data["date"] = pd.to_datetime(all_data["date"], errors="coerce")
all_data["weekday_code"] = all_data["date"].dt.weekday
all_data["week_of_year"] = all_data["date"].dt.isocalendar().week.astype(int)
all_data["month_num"] = pd.to_numeric(all_data["month"], errors="coerce")

all_data["t"] = all_data.groupby("dept_id").cumcount() + 1

# lag / rolling based on ACTUAL total_sales over full train+test validation panel
all_data["lag_1"] = all_data.groupby("dept_id")["total_sales"].shift(1)
all_data["lag_7"] = all_data.groupby("dept_id")["total_sales"].shift(7)
all_data["lag_14"] = all_data.groupby("dept_id")["total_sales"].shift(14)
all_data["lag_28"] = all_data.groupby("dept_id")["total_sales"].shift(28)

all_data["rolling_7"] = (
    all_data.groupby("dept_id")["total_sales"]
    .shift(1)
    .groupby(all_data["dept_id"])
    .rolling(7)
    .mean()
    .reset_index(level=0, drop=True)
)

all_data["rolling_14"] = (
    all_data.groupby("dept_id")["total_sales"]
    .shift(1)
    .groupby(all_data["dept_id"])
    .rolling(14)
    .mean()
    .reset_index(level=0, drop=True)
)

all_data["rolling_28"] = (
    all_data.groupby("dept_id")["total_sales"]
    .shift(1)
    .groupby(all_data["dept_id"])
    .rolling(28)
    .mean()
    .reset_index(level=0, drop=True)
)

all_data["log_lag_1"] = np.log1p(all_data["lag_1"])
all_data["log_lag_7"] = np.log1p(all_data["lag_7"])
all_data["log_price"] = np.log1p(all_data["avg_price"])

train_final = all_data[all_data["dataset"] == "train"].copy()
test_final  = all_data[all_data["dataset"] == "test"].copy()

dept_list = sorted(train_final["dept_id"].dropna().unique())

print("Train rows:", len(train_final))
print("Test rows :", len(test_final))
print("Departments:", len(dept_list))

# =========================================================
# 7. MODEL 1 — OLS
# =========================================================
print_step("Running OLS...")

ols_preds = []

from tqdm import tqdm
for dept in tqdm(dept_list, desc="Preparing DeepAR data"):
    tr = train_final[train_final["dept_id"] == dept].copy()
    te = test_final[test_final["dept_id"] == dept].copy()

    tr = tr.dropna(subset=["total_sales", "log_price", "log_lag_1", "log_lag_7", "t", "weekday", "month"])
    te = te.dropna(subset=["total_sales", "log_price", "log_lag_1", "log_lag_7", "t", "weekday", "month"])

    if len(tr) < 30 or len(te) == 0:
        continue

    tr["log_sales"] = np.log1p(tr["total_sales"])

    formula = "log_sales ~ log_price + C(weekday) + C(month) + is_event + t + log_lag_1 + log_lag_7"

    try:
        model = smf.ols(formula=formula, data=tr).fit()
        pred_log = model.predict(te)
        pred_sales = np.maximum(np.expm1(pred_log), 0)

        out = te[["date", "dept_id", "total_sales"]].copy()
        out["pred_ols"] = pred_sales
        ols_preds.append(out)
    except Exception:
        continue

ols_pred_df = pd.concat(ols_preds, ignore_index=True) if len(ols_preds) > 0 else pd.DataFrame(
    columns=["date", "dept_id", "total_sales", "pred_ols"]
)

print("OLS prediction rows:", len(ols_pred_df))

# =========================================================
# 8. MODEL 2 — RANDOM FOREST
# =========================================================
print_step("Running Random Forest...")

rf_features = [
    "avg_price", "lag_1", "lag_7", "lag_28",
    "rolling_7", "rolling_28",
    "weekday_code", "month_num", "week_of_year", "is_event", "t"
]

rf_preds = []

for dept in dept_list:
    tr = train_final[train_final["dept_id"] == dept].copy()
    te = test_final[test_final["dept_id"] == dept].copy()

    tr = tr.dropna(subset=rf_features + ["total_sales"])
    te = te.dropna(subset=rf_features + ["total_sales"])

    if len(tr) < 30 or len(te) == 0:
        continue

    X_train = tr[rf_features]
    y_train = tr["total_sales"]
    X_test = te[rf_features]

    try:
        model = RandomForestRegressor(
            n_estimators=400,
            max_features=4,
            min_samples_leaf=5,
            max_depth=10,
            random_state=123,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        pred = np.maximum(model.predict(X_test), 0)

        out = te[["date", "dept_id", "total_sales"]].copy()
        out["pred_rf"] = pred
        rf_preds.append(out)
    except Exception:
        continue

rf_pred_df = pd.concat(rf_preds, ignore_index=True) if len(rf_preds) > 0 else pd.DataFrame(
    columns=["date", "dept_id", "total_sales", "pred_rf"]
)

print("RF prediction rows:", len(rf_pred_df))

# =========================================================
# 9. MODEL 3 — XGBOOST
# =========================================================
print_step("Running XGBoost...")

xgb_preds = []

if HAS_XGB:
    xgb_features = [
        "avg_price", "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_7", "rolling_14", "rolling_28",
        "weekday_code", "month_num", "week_of_year", "is_event", "t"
    ]

    for dept in dept_list:
        tr = train_final[train_final["dept_id"] == dept].copy()
        te = test_final[test_final["dept_id"] == dept].copy()

        tr = tr.dropna(subset=xgb_features + ["total_sales"])
        te = te.dropna(subset=xgb_features + ["total_sales"])

        if len(tr) < 50 or len(te) == 0:
            continue

        X_train = tr[xgb_features]
        y_train = np.log1p(tr["total_sales"])
        X_test = te[xgb_features]

        try:
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                n_estimators=500,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                gamma=0.2,
                random_state=123
            )
            model.fit(X_train, y_train)
            pred_log = model.predict(X_test)
            pred = np.maximum(np.expm1(pred_log), 0)

            out = te[["date", "dept_id", "total_sales"]].copy()
            out["pred_xgb"] = pred
            xgb_preds.append(out)
        except Exception:
            continue

xgb_pred_df = pd.concat(xgb_preds, ignore_index=True) if len(xgb_preds) > 0 else pd.DataFrame(
    columns=["date", "dept_id", "total_sales", "pred_xgb"]
)

print("XGB prediction rows:", len(xgb_pred_df))

# =========================================================
# 10. MODEL 4 — DEEPAR
# =========================================================
print_step("Running DeepAR...")

deep_pred_df = pd.DataFrame(columns=["date", "dept_id", "total_sales", "pred_deepar"])

if HAS_DEEPAR:
    try:
        prediction_length = test_final["date"].nunique()
        train_series = []
        test_meta = []

        for dept in dept_list:
            tr = train_final[train_final["dept_id"] == dept].sort_values("date").copy()
            te = test_final[test_final["dept_id"] == dept].sort_values("date").copy()

            if len(tr) < 30 or len(te) == 0:
                continue

            train_series.append({
                "start": tr["date"].min(),
                "target": tr["total_sales"].astype(float).values,
                "item_id": str(dept)
            })

            test_meta.append({
                "dept_id": dept,
                "dates": te["date"].values,
                "actual": te["total_sales"].astype(float).values
            })

        if len(train_series) > 0:
            train_ds = ListDataset(train_series, freq="D")

            estimator = DeepAREstimator(
                prediction_length=prediction_length,
                freq="D",
                context_length=7,
                num_layers=1,
                hidden_size=5,
                dropout_rate=0.1,
                trainer_kwargs={
                    "max_epochs": 15,
                    "accelerator": "cpu",
                    "devices": 1,
                    "enable_progress_bar": True,
                    "logger": False,
                    "enable_model_summary": False
                }
            )
            print("DeepAR training started... (this may take a few minutes)")
            predictor = estimator.train(train_ds)
            forecasts = list(predictor.predict(train_ds))

            out_list = []
            for meta, fcst in tqdm(zip(test_meta, forecasts), total=len(test_meta), desc="DeepAR predicting"):
                pred_vals = np.maximum(fcst.mean[:len(meta["dates"])], 0)
                temp = pd.DataFrame({
                    "date": pd.to_datetime(meta["dates"]),
                    "dept_id": meta["dept_id"],
                    "total_sales": meta["actual"],
                    "pred_deepar": pred_vals
                })
                out_list.append(temp)

            if len(out_list) > 0:
                deep_pred_df = pd.concat(out_list, ignore_index=True)

    except Exception as e:
        print("DeepAR failed during training/prediction.")
        print("Reason:", str(e))

print("DeepAR prediction rows:", len(deep_pred_df))

# =========================================================
# 11. MERGE ALL PREDICTIONS
# =========================================================
print_step("Merging all model predictions...")

base_actual = test_final[["date", "dept_id", "total_sales"]].copy()
base_actual = base_actual.drop_duplicates(subset=["date", "dept_id"]).reset_index(drop=True)

base_actual = ensure_datetime(base_actual, "date")
ols_pred_df = ensure_datetime(ols_pred_df, "date")
rf_pred_df = ensure_datetime(rf_pred_df, "date")
xgb_pred_df = ensure_datetime(xgb_pred_df, "date")
deep_pred_df = ensure_datetime(deep_pred_df, "date")

pred_all = base_actual.merge(
    ols_pred_df[["date", "dept_id", "pred_ols"]],
    on=["date", "dept_id"],
    how="left"
).merge(
    rf_pred_df[["date", "dept_id", "pred_rf"]],
    on=["date", "dept_id"],
    how="left"
).merge(
    xgb_pred_df[["date", "dept_id", "pred_xgb"]],
    on=["date", "dept_id"],
    how="left"
).merge(
    deep_pred_df[["date", "dept_id", "pred_deepar"]],
    on=["date", "dept_id"],
    how="left"
)

print("Merged rows:", len(pred_all))
print("Non-null OLS   :", pred_all["pred_ols"].notna().sum())
print("Non-null RF    :", pred_all["pred_rf"].notna().sum())
print("Non-null XGB   :", pred_all["pred_xgb"].notna().sum())
print("Non-null DeepAR:", pred_all["pred_deepar"].notna().sum())

# =========================================================
# 12. AGGREGATE TO TOTAL SALES BY DATE
# =========================================================
print_step("Aggregating total sales by date...")

total_pred = (
    pred_all
    .groupby("date", as_index=False)
    .agg(
        actual_sales=("total_sales", "sum"),
        pred_ols=("pred_ols", "sum"),
        pred_rf=("pred_rf", "sum"),
        pred_xgb=("pred_xgb", "sum"),
        pred_deepar=("pred_deepar", "sum")
    )
    .sort_values("date")
)

# if a whole column was missing, sum() may turn it into 0; fix with all-null check
for col in ["pred_ols", "pred_rf", "pred_xgb", "pred_deepar"]:
    if pred_all[col].notna().sum() == 0:
        total_pred[col] = np.nan

print(total_pred.head())
print("Total rows in plot table:", len(total_pred))

# =========================================================
# 13. PERFORMANCE
# =========================================================
print_step("Model performance on aggregated total sales...")

ols_rmse = safe_sum_rmse(total_pred, "actual_sales", "pred_ols")
rf_rmse = safe_sum_rmse(total_pred, "actual_sales", "pred_rf")
xgb_rmse = safe_sum_rmse(total_pred, "actual_sales", "pred_xgb")
deep_rmse = safe_sum_rmse(total_pred, "actual_sales", "pred_deepar")

print("OLS RMSE     :", ols_rmse)
print("RF RMSE      :", rf_rmse)
print("XGB RMSE     :", xgb_rmse)
print("DeepAR RMSE  :", deep_rmse)

# =========================================================
# 14. PLOT
# =========================================================
print_step("Plotting total sales...")

plt.figure(figsize=(16, 8))

plt.plot(
    total_pred["date"],
    total_pred["actual_sales"],
    linewidth=3,
    label="Actual"
)

if total_pred["pred_ols"].notna().sum() > 0:
    plt.plot(
        total_pred["date"],
        total_pred["pred_ols"],
        linewidth=2,
        linestyle="--",
        label="OLS"
    )

if total_pred["pred_rf"].notna().sum() > 0:
    plt.plot(
        total_pred["date"],
        total_pred["pred_rf"],
        linewidth=2,
        linestyle="--",
        label="Random Forest"
    )

if total_pred["pred_xgb"].notna().sum() > 0:
    plt.plot(
        total_pred["date"],
        total_pred["pred_xgb"],
        linewidth=2,
        linestyle="--",
        label="XGBoost"
    )

if total_pred["pred_deepar"].notna().sum() > 0:
    plt.plot(
        total_pred["date"],
        total_pred["pred_deepar"],
        linewidth=2,
        linestyle="--",
        label="DeepAR"
    )

plt.title("Total Sales: Actual vs Predicted by Models", fontsize=20, fontweight="bold")
plt.xlabel("Date", fontsize=14)
plt.ylabel("Total Sales", fontsize=14)
plt.xticks(rotation=30)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# 15. SAVE
# =========================================================
print_step("Saving outputs...")

pred_all.to_csv(OUT_DEPT, index=False)
total_pred.to_csv(OUT_TOTAL, index=False)

print("Saved files:")
print(" -", OUT_DEPT)
print(" -", OUT_TOTAL)
print(" -", OUT_PLOT)

print_step("Done.")