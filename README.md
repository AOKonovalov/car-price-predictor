# 🚗 Car Price Predictor

Интерактивное приложение для предсказания цены подержанного автомобиля. Учебный проект по курсу «Введение в машинное обучение» НИУ ВШЭ (АБД, 2025/26).

Под капотом — `Ridge`-регрессия в лог-пространстве цены, OneHotEncoder по бренду и категориальным признакам, ручной feature engineering (`age`, `power_per_litre`, `km_per_year`, `year_sq`).

| Метрика | train | test |
|---|---|---|
| R² | 0.915 | **0.925** |
| Бизнес-метрика (доля прогнозов в ±10%) | — | **36.7%** |

## Что внутри

| Файл | Описание |
|---|---|
| `app.py` | Streamlit-приложение: EDA, ручной ввод и CSV-инференс, веса модели |
| `model.pkl` | Сериализованный пайплайн (Ridge + StandardScaler + OneHotEncoder + медианы + метаданные) |
| `train_and_save_model.py` | Скрипт, который заново обучает модель и пересохраняет `model.pkl` |
| `notebook.ipynb` | Ноутбук со всеми экспериментами: EDA, препроцессинг, сравнение моделей |
| `requirements.txt` | Зависимости (с пинами под Streamlit Cloud) |
| `runtime.txt` | Версия Python для деплоя |
| `.streamlit/config.toml` | Тема приложения |

## 🛠 Локальный запуск

```bash
git clone https://github.com/<ваш-логин>/car-price-predictor.git
cd car-price-predictor
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Откроется http://localhost:8501.

## 🚀 Как задеплоить на Streamlit Community Cloud

1. **Создай репозиторий на GitHub** (публичный — Streamlit Cloud требует доступ).
   ```bash
   cd car-price-predictor
   git init
   git add .
   git commit -m "Initial commit: car price predictor"
   git branch -M main
   git remote add origin https://github.com/<ваш-логин>/car-price-predictor.git
   git push -u origin main
   ```

2. **Зайди на [share.streamlit.io](https://share.streamlit.io)** и залогинься через GitHub.

3. Нажми **«New app»** → **«Deploy a public app from GitHub»**.

4. Заполни форму:
   * Repository: `<ваш-логин>/car-price-predictor`
   * Branch: `main`
   * Main file path: `app.py`
   * (опционально) App URL: придумай красивый поддомен, например `car-price-hse.streamlit.app`

5. Жми **«Deploy»** — первый билд занимает 2–4 минуты, дальше деплой автоматический при пуше в `main`.

6. Скопируй ссылку на работающее приложение и **впиши её сюда же**, в раздел «Демо» ниже, — она нужна для сдачи задания.

## 🔗 Демо

* Развёрнутое приложение: <https://car-price-hse.streamlit.app>  ← замени на свою ссылку

## Что делает приложение

### 📊 EDA
Гистограммы цены (обычная и логарифмическая), scatter-плот цены против выбранного признака, boxplot цены по категориям, корреляционная матрица.

### 🔮 Прогноз цены
* **Ручной ввод** — выпадающие списки для категорий + числовые поля. Дефолты подтянуты из медиан train.
* **CSV-загрузка** — загружаешь CSV того же формата, что у `cars_test.csv`, получаешь колонку `predicted_price` и можешь скачать результат. Если в CSV есть `selling_price`, дополнительно покажутся R² и доля ±10%.

### 📈 Веса модели
Топ-N признаков по модулю коэффициента (зелёный — поднимает цену, красный — опускает). Слайдер для N. Полная таблица всех коэффициентов.

## 🧪 Переобучение модели

Если хочешь пересобрать `model.pkl` с нуля:

```bash
python train_and_save_model.py
```

Скрипт сам скачает данные с `github.com/evgpat/datasets`, обработает их, подберёт `alpha` через `GridSearchCV` (10 фолдов) и сохранит новый артефакт.

## Структура проекта

```
car-price-predictor/
├── app.py                  # Streamlit-приложение
├── model.pkl               # обученный пайплайн
├── train_and_save_model.py # скрипт обучения
├── notebook.ipynb          # ноутбук с экспериментами
├── requirements.txt
├── runtime.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
└── README.md
```

## Замечания

* Модель училась на индийском датасете и предсказывает цену в индийских рупиях. Для другой валюты нужно переобучать модель.
* Возраст машины считается как `2025 - year` — для долгосрочного использования это значение в коде стоит сделать динамическим.
* Признак `name` агрегирован до бренда (первое слово в названии), чтобы количество категорий не разрасталось.
