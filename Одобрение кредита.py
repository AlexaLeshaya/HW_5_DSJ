import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions

st.title("👨‍💼💸👩‍💼Анализ заявок на кредит")
st.write(
    """
    Это Streamlit‑приложение реализует обработку и анализ набора данных "Одобрение кредита" (UCI).
    В приложении производится загрузка, предобработка данных, отбор признаков, обучение нескольких моделей,
    визуализация 3D распределения, границ решений и ROC-кривых. Интерактивные элементы позволяют менять параметры моделей.
    """
)

# Боковая панель с настройками
st.sidebar.header("Настройки и параметры")

# Параметры для разделения выборки
test_size = st.sidebar.slider("Доля тестовой выборки (test_size)", 0.1, 0.2, 0.3, 0.15)

# Параметры моделей
k_neighbors = st.sidebar.slider("Количество соседей для kNN", 1, 15, 3, 1)
max_depth = st.sidebar.slider("Максимальная глубина дерева", 1, 20, 5, 1)
max_iter = st.sidebar.slider("Максимальное число итераций (Logistic Regression)", 100, 1000, 565, 50)

# Функция загрузки и первичной обработки данных
@st.cache_data
def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
    df = pd.read_csv(url, header=None, na_values='?')
    # Удаляем строки с пропущенными значениями в таргете (столбец 15)
    df = df.dropna(subset=[15])
    # Преобразуем все объектные столбцы (кроме таргета) в числовые, если возможно
    for col in df.columns:
        if col != 15 and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Удаляем столбцы, состоящие только из NaN
    df.dropna(axis=1, how='all', inplace=True)
    # Заполняем пропуски в числовых признаках по группе таргета
    num_cols = [col for col in df.columns if col != 15 and pd.api.types.is_numeric_dtype(df[col])]
    df[num_cols] = df.groupby(15)[num_cols].transform(lambda x: x.fillna(x.mean()))
    # Преобразуем таргет: '+' → 1, '-' → 0
    df[15] = df[15].map({'+': 1, '-': 0})
    return df

data = load_data()

# Отображение данных
st.subheader("Обзор данных")
st.write(f"Размер набора данных: {data.shape}")
st.dataframe(data.head(10))

# Отбор признаков по корреляции (из числовых, с более чем 10 уникальных значений)
num_features = [col for col in data.columns if col != 15 and data[col].nunique() > 10]
corr = {col: data[col].corr(data[15]) for col in num_features}
sorted_features = sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)
top3_features = sorted_features[:3]

st.write("**Три наиболее значимых признака (по корреляции с таргетом):**")
for feature, corr_value in top3_features:
    st.write(f"Признак {feature}: корреляция {corr_value:.3f}")

# Интерактивный выбор двух признаков для классификации
st.sidebar.subheader("Выбор признаков для классификации")
numeric_cols = [col for col in data.columns if col != 15 and pd.api.types.is_numeric_dtype(data[col])]
default_feats = [10, 7] if 10 in numeric_cols and 7 in numeric_cols else numeric_cols[:2]
selected_features = st.sidebar.multiselect(
    "Выберите ровно 2 признака для классификации",
    options=numeric_cols,
    default=default_feats
)
if len(selected_features) != 2:
    st.error("Пожалуйста, выберите ровно 2 признака!")
    st.stop()

# Подготовка данных для классификации
X = data[selected_features]
y = data[15]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение моделей
st.subheader("Обучение моделей")
models = [
    ("kNN", KNeighborsClassifier(n_neighbors=k_neighbors)),
    ("Logistic Regression", LogisticRegression(max_iter=max_iter)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=max_depth, random_state=42))
]

results = {}
for name, model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    # ROC-кривая (если модель поддерживает predict_proba)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc = None, None, None
    results[name] = {
        "model": model,
        "accuracy": acc,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc
    }
    st.write(f"**{name}** — Accuracy: {acc:.3f}, AUC: {roc_auc:.3f}" if roc_auc else f"**{name}** — Accuracy: {acc:.3f}")

# Отображение подробного отчёта по моделям
with st.expander("Показать детальный classification report для всех моделей"):
    for name, res in results.items():
        st.write(f"### {name}")
        st.text(classification_report(y_test, res["model"].predict(X_test_scaled)))

# Визуализация 3D распределения по трём наиболее значимым признакам
st.subheader("3D визуализация (топ-3 признака по корреляции)")
features_3d = [feature for feature, _ in top3_features]
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
markers = {1: 'o', 0: '^'}
colors  = {1: 'blue', 0: 'red'}
for cls in data[15].unique():
    data_cls = data[data[15] == cls]
    ax.scatter(
        data_cls[features_3d[0]],
        data_cls[features_3d[1]],
        data_cls[features_3d[2]],
        marker=markers[cls],
        color=colors[cls],
        label=f"Класс {cls}"
    )
ax.set_title("3D-график: Одобрение кредита (UCI)")
ax.set_xlabel(f"Признак {features_3d[0]}")
ax.set_ylabel(f"Признак {features_3d[1]}")
ax.set_zlabel(f"Признак {features_3d[2]}")
ax.legend()
st.pyplot(fig)

# Визуализация границ решений для выбранных 2 признаков
st.subheader("Границы решений моделей (2D)")
fig2, axes = plt.subplots(3, 1, figsize=(8, 16))
for i, (name, model) in enumerate(models):
    ax = axes[i]
    plot_decision_regions(X_train_scaled, y_train.values, clf=model, legend=2, ax=ax)
    ax.set_xlabel(str(selected_features[0]))
    ax.set_ylabel(str(selected_features[1]))
    ax.set_title(f"Границы решений: {name}")
plt.tight_layout()
st.pyplot(fig2)

# Построение ROC-кривых
st.subheader("ROC-кривые для моделей")
fig3, ax3 = plt.subplots(figsize=(8, 6))
for name, res in results.items():
    if res["roc_auc"] is not None:
        ax3.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['roc_auc']:.3f})")
ax3.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Случайное угадывание')
ax3.set_title("ROC-кривые (тестовая выборка)")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend(loc="lower right")
st.pyplot(fig3)
