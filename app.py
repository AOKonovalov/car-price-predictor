"""Streamlit-приложение для предсказания цены автомобиля.

Запуск:  streamlit run app.py
"""
import io
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------- 0. Настройки страницы ----------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)


# ---------- 1. Загрузка модели ----------
@st.cache_resource
def load_artifact():
    path = Path(__file__).parent / "model.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


artifact = load_artifact()
model = artifact["model"]
scaler = artifact["scaler"]
ohe = artifact["ohe"]
medians = artifact["medians"]
num_cols_final = artifact["num_cols_final"]
cat_cols = artifact["cat_cols"]
fe_cols = artifact["fe_cols"]
feature_names = artifact["feature_names"]
metrics = artifact["metrics"]
categories = artifact["categories"]
defaults = artifact["feature_defaults"]
sample_df = artifact["sample_data"]


# ---------- 2. Препроцессинг (тот же, что был при обучении) ----------
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
    if df["mileage"].dtype == object:
        df["mileage"] = df["mileage"].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
    if df["engine"].dtype == object:
        df["engine"] = df["engine"].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
    if df["max_power"].dtype == object:
        df["max_power"] = df["max_power"].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
    if "torque" in df.columns and df["torque"].dtype == object:
        parsed = df["torque"].apply(parse_torque)
        df["torque"] = parsed.apply(lambda t: t[0])
        df["max_torque_rpm"] = parsed.apply(lambda t: t[1])
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Применяет тот же препроцессинг, что и при обучении."""
    df = df.copy()

    # парсинг строковых единиц
    df = clean_units(df)

    # бренд из name
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.split().str[0]

    # заполнение пропусков медианами из train
    for col, med in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)

    # типы
    df["engine"] = df["engine"].astype(int)
    df["seats"] = df["seats"].astype(int)

    # feature engineering
    df["age"] = 2025 - df["year"]
    df["power_per_litre"] = df["max_power"] / (df["engine"] / 1000)
    df["km_per_year"] = df["km_driven"] / df["age"].clip(lower=1)
    df["year_sq"] = df["year"] ** 2

    # OHE
    ohe_arr = ohe.transform(df[cat_cols])
    ohe_df = pd.DataFrame(
        ohe_arr,
        columns=ohe.get_feature_names_out(cat_cols),
        index=df.index,
    )
    numerics = df[num_cols_final + fe_cols].reset_index(drop=True)
    numerics_scaled = pd.DataFrame(
        scaler.transform(numerics),
        columns=numerics.columns,
    )
    X = pd.concat(
        [numerics_scaled.reset_index(drop=True), ohe_df.reset_index(drop=True)],
        axis=1,
    )
    return X[feature_names]


def predict(df: pd.DataFrame) -> np.ndarray:
    X = preprocess(df)
    pred_log = model.predict(X)
    return np.expm1(pred_log)


# ---------- 3. Сайдбар ----------
st.sidebar.title("🚗 Car Price Predictor")
st.sidebar.caption("Ridge-регрессия + категориальные фичи + feature engineering")

page = st.sidebar.radio(
    "Раздел",
    ["📊 EDA", "🔮 Прогноз цены", "📈 Веса модели"],
)

with st.sidebar.expander("Метрики модели"):
    st.metric("R² на test", f"{metrics['r2_test']:.4f}")
    st.metric("R² на train", f"{metrics['r2_train']:.4f}")
    st.metric("Доля ±10% (test)", f"{metrics['business_metric_test']*100:.1f}%")
    st.caption(f"Лучший α: {artifact['best_alpha']:.4f}")


# ============================================================
#                          EDA
# ============================================================
if page == "📊 EDA":
    st.title("📊 EDA по тренировочной выборке")
    st.write(
        "Случайная подвыборка из тренировочных данных "
        f"({len(sample_df)} строк). Здесь можно посмотреть распределения и связи признаков."
    )

    with st.expander("Превью данных"):
        st.dataframe(sample_df.head(20), use_container_width=True)

    st.subheader("Распределение целевой переменной")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(sample_df["selling_price"], bins=50, ax=ax, color="#3b82f6")
        ax.set_title("selling_price")
        ax.set_xlabel("Цена")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(np.log1p(sample_df["selling_price"]), bins=50, ax=ax, color="#f97316")
        ax.set_title("log(1 + selling_price)")
        ax.set_xlabel("log-цена")
        st.pyplot(fig)

    st.info(
        "У распределения цен тяжёлый правый хвост — несколько очень дорогих машин. "
        "После лог-преобразования распределение становится почти симметричным, "
        "поэтому модель учится в лог-пространстве."
    )

    st.subheader("Зависимости цены от ключевых признаков")
    feat_x = st.selectbox(
        "Признак на оси X",
        ["year", "km_driven", "mileage", "engine", "max_power", "torque", "seats"],
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=sample_df, x=feat_x, y="selling_price", alpha=0.3, ax=ax)
    ax.set_yscale("log")
    ax.set_title(f"selling_price vs {feat_x} (Y в лог-шкале)")
    st.pyplot(fig)

    st.subheader("Цена по категориям")
    cat = st.selectbox(
        "Категориальный признак",
        ["fuel", "transmission", "seller_type", "owner"],
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=sample_df, x=cat, y="selling_price", ax=ax)
    ax.set_yscale("log")
    ax.set_title(f"selling_price по {cat}")
    plt.xticks(rotation=15)
    st.pyplot(fig)

    st.subheader("Корреляции (числовые признаки)")
    num_for_corr = [
        "year", "km_driven", "mileage", "engine",
        "max_power", "torque", "max_torque_rpm", "seats", "selling_price",
    ]
    corr = sample_df[num_for_corr].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)


# ============================================================
#                       PREDICTION
# ============================================================
elif page == "🔮 Прогноз цены":
    st.title("🔮 Прогноз цены автомобиля")

    tab_manual, tab_csv = st.tabs(["Ручной ввод", "Загрузить CSV"])

    # ---- Ручной ввод ----
    with tab_manual:
        st.write("Заполните характеристики автомобиля:")

        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.selectbox("Бренд (name)", sorted(categories["name"]),
                                index=sorted(categories["name"]).index("Maruti")
                                if "Maruti" in categories["name"] else 0)
            fuel = st.selectbox("Тип топлива", categories["fuel"])
            seller_type = st.selectbox("Тип продавца", categories["seller_type"])
            transmission = st.selectbox("Трансмиссия", categories["transmission"])
            owner = st.selectbox("Владельцы", categories["owner"])
            seats = st.selectbox("Количество мест",
                                 [int(s) for s in sorted(categories["seats"])],
                                 index=2)

        with col2:
            year = st.number_input("Год выпуска", min_value=1990, max_value=2025,
                                   value=int(defaults["year"]))
            km_driven = st.number_input("Пробег, км", min_value=0, max_value=2_000_000,
                                        value=int(defaults["km_driven"]), step=1000)
            mileage = st.number_input("Расход (kmpl)", min_value=0.0, max_value=50.0,
                                      value=float(defaults["mileage"]), step=0.5)
            engine = st.number_input("Объём двигателя, см³", min_value=500, max_value=10000,
                                     value=int(defaults["engine"]), step=50)

        with col3:
            max_power = st.number_input("Мощность, л.с.", min_value=10.0, max_value=1000.0,
                                        value=float(defaults["max_power"]), step=5.0)
            torque = st.number_input("Крутящий момент, Nm", min_value=10.0, max_value=2000.0,
                                     value=float(defaults["torque"]), step=10.0)
            max_torque_rpm = st.number_input("Обороты макс. момента, rpm",
                                             min_value=500.0, max_value=10000.0,
                                             value=float(defaults["max_torque_rpm"]),
                                             step=100.0)

        if st.button("💰 Узнать цену", type="primary"):
            row = pd.DataFrame([{
                "name": name,
                "year": year,
                "km_driven": km_driven,
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner,
                "mileage": mileage,
                "engine": engine,
                "max_power": max_power,
                "torque": torque,
                "max_torque_rpm": max_torque_rpm,
                "seats": seats,
            }])

            pred = predict(row)[0]
            st.success(f"### Предсказанная цена: **{pred:,.0f}**")

            with st.expander("Признаки, которые пошли в модель"):
                st.json(row.iloc[0].to_dict())

    # ---- CSV ----
    with tab_csv:
        st.write(
            "Загрузите CSV с теми же столбцами, что в исходных данных. "
            "Столбец `selling_price` (если есть) будет проигнорирован."
        )
        uploaded = st.file_uploader("CSV-файл", type=["csv"])

        if uploaded is not None:
            df_in = pd.read_csv(uploaded)
            st.write(f"Загружено строк: **{len(df_in)}**")
            st.dataframe(df_in.head(10), use_container_width=True)

            required = ["name", "year", "km_driven", "fuel", "seller_type",
                        "transmission", "owner", "mileage", "engine",
                        "max_power", "torque", "seats"]
            missing = [c for c in required if c not in df_in.columns]
            if missing:
                st.error(f"В файле не хватает столбцов: {missing}")
            else:
                try:
                    preds = predict(df_in)
                    out = df_in.copy()
                    out["predicted_price"] = preds.round(0).astype(int)

                    st.success(f"Готово! Предсказали цены для {len(out)} объектов.")
                    st.dataframe(out, use_container_width=True)

                    buf = io.BytesIO()
                    out.to_csv(buf, index=False)
                    st.download_button(
                        "📥 Скачать результат",
                        data=buf.getvalue(),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

                    # если есть истинная цена — сравним
                    if "selling_price" in df_in.columns:
                        from sklearn.metrics import r2_score
                        r2 = r2_score(df_in["selling_price"], preds)
                        diff = np.abs(preds - df_in["selling_price"]) / df_in["selling_price"]
                        st.info(
                            f"**Если столбец `selling_price` корректный:** "
                            f"R² = {r2:.4f}, доля прогнозов в ±10%: "
                            f"{(diff <= 0.10).mean()*100:.1f}%"
                        )
                except Exception as e:
                    st.error(f"Ошибка при инференсе: {e}")


# ============================================================
#                       WEIGHTS
# ============================================================
elif page == "📈 Веса модели":
    st.title("📈 Веса обученной модели")
    st.write(
        f"Модель: **Ridge(alpha={artifact['best_alpha']:.4f})** в лог-пространстве цены. "
        f"Всего признаков после OHE и FE: **{len(feature_names)}**."
    )

    coefs = pd.DataFrame({
        "feature": feature_names,
        "coef": model.coef_,
        "abs_coef": np.abs(model.coef_),
    }).sort_values("abs_coef", ascending=False)

    st.subheader("Топ-25 признаков по модулю коэффициента")
    top_n = st.slider("Сколько показать", min_value=5, max_value=min(50, len(coefs)),
                      value=25)
    top = coefs.head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    colors = ["#22c55e" if c > 0 else "#ef4444" for c in top["coef"]]
    ax.barh(top["feature"][::-1], top["coef"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Коэффициент (в лог-пространстве цены)")
    ax.set_title(f"Top-{top_n} признаков по |coef|")
    st.pyplot(fig)

    st.caption(
        "Зелёный — увеличивает прогнозируемую цену, красный — уменьшает. "
        "Коэффициенты сравнимы между собой, потому что числовые признаки стандартизованы."
    )

    st.subheader("Все коэффициенты (таблица)")
    st.dataframe(coefs.reset_index(drop=True), use_container_width=True, height=400)
