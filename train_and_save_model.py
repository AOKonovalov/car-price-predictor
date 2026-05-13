"""Обучает финальную модель (Ridge + категориальные + FE + log-таргет)
и сохраняет пайплайн целиком в model.pkl.
"""
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CARS_TRAIN = "https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv"
CARS_TEST = "https://github.com/evgpat/datasets/raw/refs/heads/main/cars_test.csv"
RANDOM_STATE = 42
ARTIFACT_PATH = Path(__file__).parent / "model.pkl"


# ---------- 1. Предобработка ----------
def parse_torque(value):
    if pd.isna(value):
        return np.nan, np.nan
    s = str(value).lower().replace(",", "")
    nums = re.findall(r"\d+\.?\d*", s)
    if not nums:
        return np.nan, np.nan
    torque_val = float(nums[0])
    if "kgm" in s:
        torque_val *= 9.80665
    max_rpm = float(nums[-1]) if len(nums) >= 2 else np.nan
    return torque_val, max_rpm


def clean_units(df):
    df = df.copy()
    df["mileage"] = df["mileage"].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
    df["engine"] = df["engine"].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
    df["max_power"] = df["max_power"].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
    parsed = df["torque"].apply(parse_torque)
    df["torque"] = parsed.apply(lambda t: t[0])
    df["max_torque_rpm"] = parsed.apply(lambda t: t[1])
    return df


def add_features(df):
    df = df.copy()
    df["age"] = 2025 - df["year"]
    df["power_per_litre"] = df["max_power"] / (df["engine"] / 1000)
    df["km_per_year"] = df["km_driven"] / df["age"].clip(lower=1)
    df["year_sq"] = df["year"] ** 2
    return df


# ---------- 2. Загрузка и подготовка ----------
print("Загружаем данные…")
df_train = pd.read_csv(CARS_TRAIN)
df_test = pd.read_csv(CARS_TEST)

# дедупликация по признаковому описанию
feature_cols = [c for c in df_train.columns if c != "selling_price"]
df_train = df_train.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)

# парсинг единиц измерения
df_train = clean_units(df_train)
df_test = clean_units(df_test)

# медианы — только по train
num_cols_with_nan = ["mileage", "engine", "max_power", "torque", "max_torque_rpm", "seats"]
medians = df_train[num_cols_with_nan].median()
df_train[num_cols_with_nan] = df_train[num_cols_with_nan].fillna(medians)
df_test[num_cols_with_nan] = df_test[num_cols_with_nan].fillna(medians)

for df in (df_train, df_test):
    df["engine"] = df["engine"].astype(int)
    df["seats"] = df["seats"].astype(int)

# бренд из name
df_train["name"] = df_train["name"].str.split().str[0]
df_test["name"] = df_test["name"].str.split().str[0]

# ---------- 3. OneHot + StandardScaler + FE ----------
cat_cols = ["name", "fuel", "seller_type", "transmission", "owner", "seats"]
num_cols_final = ["year", "km_driven", "mileage", "engine",
                  "max_power", "torque", "max_torque_rpm"]
fe_cols = ["age", "power_per_litre", "km_per_year", "year_sq"]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
ohe.fit(df_train[cat_cols])

def build_matrix(df, scaler=None, fit_scaler=False):
    df = add_features(df)
    ohe_arr = ohe.transform(df[cat_cols])
    ohe_df = pd.DataFrame(
        ohe_arr,
        columns=ohe.get_feature_names_out(cat_cols),
        index=df.index,
    )
    numerics = df[num_cols_final + fe_cols].reset_index(drop=True)
    if fit_scaler:
        scaler = StandardScaler()
        numerics_scaled = pd.DataFrame(
            scaler.fit_transform(numerics),
            columns=numerics.columns,
        )
    else:
        numerics_scaled = pd.DataFrame(
            scaler.transform(numerics),
            columns=numerics.columns,
        )
    X = pd.concat([numerics_scaled.reset_index(drop=True),
                   ohe_df.reset_index(drop=True)], axis=1)
    return X, scaler


X_train, scaler = build_matrix(df_train, fit_scaler=True)
X_test, _ = build_matrix(df_test, scaler=scaler)

y_train_log = np.log1p(df_train["selling_price"])
y_test_log = np.log1p(df_test["selling_price"])

# ---------- 4. Обучение Ridge через GridSearchCV ----------
print("Обучаем Ridge + GridSearchCV…")
gs = GridSearchCV(
    Ridge(),
    param_grid={"alpha": np.logspace(-2, 4, 25)},
    scoring="r2",
    cv=10,
    n_jobs=-1,
)
gs.fit(X_train, y_train_log)

best_model = gs.best_estimator_
print("Лучший alpha:", gs.best_params_)

# финальные метрики
from sklearn.metrics import r2_score, mean_squared_error as MSE
pred_train_orig = np.expm1(best_model.predict(X_train))
pred_test_orig = np.expm1(best_model.predict(X_test))

metrics = {
    "r2_train": float(r2_score(df_train["selling_price"], pred_train_orig)),
    "r2_test": float(r2_score(df_test["selling_price"], pred_test_orig)),
    "mse_train": float(MSE(df_train["selling_price"], pred_train_orig)),
    "mse_test": float(MSE(df_test["selling_price"], pred_test_orig)),
    "business_metric_test": float(
        np.mean(
            np.abs(pred_test_orig - df_test["selling_price"])
            / df_test["selling_price"]
            <= 0.10
        )
    ),
}
print("Метрики:", metrics)

# ---------- 5. Сохраняем артефакт ----------
artifact = {
    "model": best_model,
    "scaler": scaler,
    "ohe": ohe,
    "medians": medians.to_dict(),
    "num_cols_final": num_cols_final,
    "cat_cols": cat_cols,
    "fe_cols": fe_cols,
    "feature_names": list(X_train.columns),
    "metrics": metrics,
    "best_alpha": float(gs.best_params_["alpha"]),
    # ⬇ список известных категорий — нужно для подсказок в UI
    "categories": {c: list(ohe.categories_[i]) for i, c in enumerate(cat_cols)},
    # ⬇ примеры значений для дефолтов в UI
    "feature_defaults": {
        "year": int(df_train["year"].median()),
        "km_driven": int(df_train["km_driven"].median()),
        "mileage": float(df_train["mileage"].median()),
        "engine": int(df_train["engine"].median()),
        "max_power": float(df_train["max_power"].median()),
        "torque": float(df_train["torque"].median()),
        "max_torque_rpm": float(df_train["max_torque_rpm"].median()),
        "seats": int(df_train["seats"].median()),
    },
    # ⬇ компактные данные для EDA-вкладки приложения
    "sample_data": df_train.sample(
        min(2000, len(df_train)), random_state=RANDOM_STATE
    ).reset_index(drop=True),
}

with ARTIFACT_PATH.open("wb") as f:
    pickle.dump(artifact, f)

print(f"\nГотово. Артефакт сохранён в {ARTIFACT_PATH} "
      f"({ARTIFACT_PATH.stat().st_size / 1024:.1f} KB)")
