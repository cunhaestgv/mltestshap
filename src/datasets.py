"""Dataset loaders.

Every loader returns a dict with chronological (or official) train/val/test
splits, already min-max scaled with training statistics:

    X_train, y_train, X_val, y_val, X_test, y_test : numpy arrays
    feature_names : names of the flattened features (X.reshape(n, -1) order)
    task          : "regression" or "classification"
    kind          : "seq" (LSTM input, shape (n, lookback, n_vars)) or "tab"
    n_vars        : number of variables per time step (seq) or features (tab)
    var_names     : names of the variables (before unfolding over time)
    x_min, x_max  : per-variable scaling, to report boundaries in raw units
    y_min, y_max  : target scaling (regression only)
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _split_idx(n, train=0.7, val=0.15):
    a, b = int(n * train), int(n * (train + val))
    return slice(0, a), slice(a, b), slice(b, n)


def _windows(values, target, lookback):
    """X[i] = values[i-lookback+1 .. i], y[i] = target[i+1] (next step)."""
    X, y = [], []
    for i in range(lookback - 1, len(values) - 1):
        X.append(values[i - lookback + 1:i + 1])
        y.append(target[i + 1])
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")


def _seq_names(var_names, lookback):
    # time step 0 is the oldest one (t-(lookback-1)), the last one is t
    names = []
    for step in range(lookback):
        lag = lookback - 1 - step
        suffix = "_t" if lag == 0 else "_t-%d" % lag
        names += [v + suffix for v in var_names]
    return names


def _make_seq(df, var_names, target_col, lookback):
    values = df[var_names].to_numpy(dtype="float64")
    target = df[target_col].to_numpy(dtype="float64")
    tr, va, te = _split_idx(len(df))
    x_min, x_max = values[tr].min(0), values[tr].max(0)
    span = np.where(x_max > x_min, x_max - x_min, 1.0)
    y_min, y_max = target[tr].min(), target[tr].max()
    values = (values - x_min) / span
    target = (target - y_min) / (y_max - y_min)
    out = {}
    for name, sl in (("train", tr), ("val", va), ("test", te)):
        out["X_" + name], out["y_" + name] = _windows(values[sl], target[sl], lookback)
    out.update(feature_names=_seq_names(var_names, lookback), task="regression",
               kind="seq", n_vars=len(var_names), var_names=list(var_names),
               lookback=lookback, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    return out


def load_bike(lookback=3):
    """UCI Bike Sharing (hourly). Predicts the rental count of the next hour.

    instant/dteday are dropped (row index and date would leak time), and so are
    casual/registered (they sum to the target).
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "bike", "hour.csv"))
    var_names = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
                 "weathersit", "temp", "atemp", "hum", "windspeed", "cnt"]
    return _make_seq(df, var_names, "cnt", lookback)


def load_diet(lookback=3, with_date=False):
    """Kaggle weight-loss dataset (diet_data.csv), as used in the paper.

    Predicts the next-day weight in kg. Date is excluded unless with_date=True,
    because a day counter lets the model learn the time trend instead of the
    effect of the behavioural features.
    """
    path = os.path.join(DATA_DIR, "diet", "diet_data.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"weight_oz": "Weight_Kg"})
    df["Weight_Kg"] = pd.to_numeric(df["Weight_Kg"], errors="coerce") * 0.0283495231
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    var_names = ["Weight_Kg", "calories", "five_donuts", "walk", "run", "wine", "prot", "weight"]
    df = df.dropna(subset=["Date"] + var_names).reset_index(drop=True)
    df["Date"] = (df["Date"] - df["Date"].iloc[0]).dt.days
    if with_date:
        var_names = ["Date"] + var_names
    return _make_seq(df, var_names, "Weight_Kg", lookback)


ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
              "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
              "hours_per_week", "native_country", "income"]


def _adult_frame(path, skip):
    df = pd.read_csv(path, header=None, names=ADULT_COLS, skiprows=skip,
                     skipinitialspace=True, na_values="?")
    df = df.dropna().reset_index(drop=True)
    out = pd.DataFrame({
        "age": df.age,
        "education_num": df.education_num,
        "capital_gain": df.capital_gain,
        "capital_loss": df.capital_loss,
        "hours_per_week": df.hours_per_week,
        "male": (df.sex == "Male").astype(int),
        "married": df.marital_status.isin(["Married-civ-spouse", "Married-AF-spouse"]).astype(int),
        "white": (df.race == "White").astype(int),
        "us_native": (df.native_country == "United-States").astype(int),
        "self_employed": df.workclass.str.startswith("Self-emp").astype(int),
        "gov_employed": df.workclass.str.endswith("-gov").astype(int),
    })
    y = df.income.str.rstrip(".").eq(">50K").astype("float32").to_numpy()
    return out, y


def load_adult():
    """UCI Adult. Classifies whether income is above 50K (binary)."""
    tr_df, y_tr = _adult_frame(os.path.join(DATA_DIR, "adult", "adult.data"), 0)
    te_df, y_te = _adult_frame(os.path.join(DATA_DIR, "adult", "adult.test"), 1)
    var_names = list(tr_df.columns)
    X_all, X_te = tr_df.to_numpy("float64"), te_df.to_numpy("float64")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X_all))
    n_val = int(0.15 * len(X_all))
    va_idx, tr_idx = perm[:n_val], perm[n_val:]
    x_min, x_max = X_all[tr_idx].min(0), X_all[tr_idx].max(0)
    span = np.where(x_max > x_min, x_max - x_min, 1.0)
    scale = lambda a: ((a - x_min) / span).astype("float32")
    return dict(X_train=scale(X_all[tr_idx]), y_train=y_tr[tr_idx],
                X_val=scale(X_all[va_idx]), y_val=y_tr[va_idx],
                X_test=scale(X_te), y_test=y_te,
                feature_names=var_names, task="classification", kind="tab",
                n_vars=len(var_names), var_names=var_names, lookback=1,
                x_min=x_min, x_max=x_max, y_min=0.0, y_max=1.0)


LOADERS = {"bike": load_bike, "adult": load_adult, "diet": load_diet,
           "diet_date": lambda: load_diet(with_date=True)}


def raw_value(data, f, scaled):
    """Convert a scaled value of flattened feature f back to raw units."""
    v = f % data["n_vars"]
    return data["x_min"][v] + scaled * (data["x_max"][v] - data["x_min"][v])
