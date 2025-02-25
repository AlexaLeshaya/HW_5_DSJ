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

# Заголовок
st.title("👨‍💼💸👩‍💼 Анализ заявок на кредит")
st.write(
    """
    Это Streamlit-приложение для анализа одобрения кредитов. 
    Включает предобработку данных, обучение моделей, визуализацию 3D-распределения и границ решений.
    """
)

# Боковая панель с настройками
st.sidebar.header("Настройки модели")
test_size = st.sidebar.slider("Доля тестовой выборки", 0.1, 0.3, 0.2, 0.05)
k_neighbors = st.sidebar.slider("Число соседей (kNN)", 1, 15, 3, 1)
max_depth = st.sidebar.slider("Глубина дерева", 1, 20, 5, 1)
max_iter = st.sidebar.slider("Макс. итераций (Logistic Regression)", 100, 1000, 500, 5)

# Функция загрузки и обработки данных
@st.cache_data
def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
    df = pd.read_csv(url, header=None, na_values='?')

    df = df.dropna(subset=[15])  # Удаляем строки с NaN в таргете

    for col in df.columns:
        if col != 15 and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(axis=1, how='all', inplace=True)  # Удаляем пустые столбцы
    num_cols = [col for col in df.columns if col != 15 and pd.api.types.is_numeric_dtype(df[col])]
    df[num_cols] = df.groupby(15)[num_cols].transform(lambda x: x.fillna(x.mean()))  # Заполняем NaN средним по классу
    df[15] = df[15].map({'+': 1, '-': 0})  # Перекодировка таргета
    return df

data = load_data()
st.subheader("Обзор данных")
st.write(f"Размер данных: {data.shape}")
st.dataframe(data.head(10))

# Выбор 3 лучших признаков
num_features = [col for col in data.columns if col != 15 and data[col].nunique() > 10]
corr = {col: data[col].corr(data[15]) for col in num_features}
sorted_features = sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)
top3_features = sorted_features[:3]

st.write("**Три самых значимых признака (по корреляции):**")
for feature, corr_value in top3_features:
    st.write(f"Признак {feature}: {corr_value:.3f}")

# Выбор 2 признаков для классификации
st.sidebar.subheader("Выбор признаков")
numeric_cols = [col for col in data.columns if col != 15 and pd.api.types.is_numeric_dtype(data[col])]
selected_features = st.sidebar.multiselect("Выберите 2 признака", options=numeric_cols, default=numeric_cols[:2])

if len(selected_features) != 2:
    st.error("Выберите ровно 2 признака!")
    st.stop()

# Разделение данных
X = data[selected_features]
y = data[15]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
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

    # ROC-кривая
    fpr, tpr, roc_auc = None, None, None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

    results[name] = {"model": model, "accuracy": acc, "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}
    st.write(f"**{name}** — Accuracy: {acc:.3f}, AUC: {roc_auc:.3f}" if roc_auc else f"**{name}** — Accuracy: {acc:.3f}")

# Границы решений (2D)
st.subheader("Границы решений")
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

y_train_array = y_train.values  # Преобразуем в массив для plot_decision_regions
for i, (name, model) in enumerate(models):
    ax = axes[i]
    plot_decision_regions(X_train_scaled, y_train_array, clf=model, legend=2, ax=ax)
    ax.set_xlabel(str(selected_features[0]))
    ax.set_ylabel(str(selected_features[1]))
    ax.set_title(f"Границы решений: {name}")

plt.subplots_adjust(hspace=0.5)  # Вместо plt.tight_layout()
st.pyplot(fig)

# Визуализация 3D-распределения
st.subheader("3D-визуализация значимых признаков")
features_3d = [feature for feature, _ in top3_features]
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
markers = {1: 'o', 0: '^'}
colors = {1: 'blue', 0: 'red'}

for cls in data[15].unique():
    data_cls = data[data[15] == cls]
    ax_3d.scatter(
        data_cls[features_3d[0]], data_cls[features_3d[1]], data_cls[features_3d[2]],
        marker=markers[cls], color=colors[cls], label=f"Класс {cls}"
    )

ax_3d.set_title("3D-график: Одобрение кредита")
ax_3d.set_xlabel(features_3d[0])
ax_3d.set_ylabel(features_3d[1])
ax_3d.set_zlabel(features_3d[2])
ax_3d.legend()
st.pyplot(fig_3d)

# Построение ROC-кривых
st.subheader("ROC-кривые моделей")
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
for name, res in results.items():
    if res["roc_auc"] is not None and res["fpr"] is not None and res["tpr"] is not None:
        ax_roc.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['roc_auc']:.3f})")

ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Случайное угадывание')
ax_roc.set_title("ROC-кривые (тестовая выборка)")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)
