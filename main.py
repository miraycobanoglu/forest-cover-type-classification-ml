import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

sns.set(style="whitegrid")

# =====================================================
# 1) Veri yükle
# =====================================================
df = pd.read_csv("forest_cover_with_few_missing.csv")

print("\n===== EKSİK DEĞER DURUMU (DOLDURMADAN ÖNCE) =====")
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0])
print(f"\nToplam eksik değer sayısı: {df.isnull().sum().sum()}")

# -----------------------------------------------------
# HEDEF DEĞİŞKEN DAĞILIMI
# -----------------------------------------------------
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x="Cover_Type", palette="viridis")
for p in ax.patches:
    ax.annotate(
        f'{p.get_height()}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center',
        va='center',
        fontsize=10,
        xytext=(0, 7),
        textcoords='offset points'
    )
plt.title("Hedef Değişken Dağılımı (Sınıf Dengesizliği)")
plt.xlabel("Cover Type")
plt.ylabel("Örnek Sayısı")
plt.show()

# =====================================================
# 2) Target NaN olan satırları sil
# =====================================================
df = df.dropna(subset=["Cover_Type"])

# =====================================================
# 3) X / y ayır
# =====================================================
X = df.drop(columns=["Cover_Type"])
y = df["Cover_Type"]

# =====================================================
# 4) SADECE X için imputation
# =====================================================
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("\n===== EKSİK DEĞER DURUMU (DOLDURMADAN SONRA) =====")
print(f"Toplam eksik değer sayısı: {X.isnull().sum().sum()}")

# =====================================================
# 5) Train-test split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 5.1) 10-Fold Cross Validation
# =====================================================
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# =====================================================
# 6) Modeller (SVM YOK – HİBRİT VAR)
# =====================================================
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Hybrid (LogReg + RF)": VotingClassifier(
        estimators=[
            ("lr", make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)
            )),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ))
        ],
        voting="soft"
    )
}

results = []

# =====================================================
# 7) 10-Fold CV + Confusion Matrix
# =====================================================
for name, model in models.items():
    print(f"\n----- {name} -----")

    acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_tr, y_tr)
        y_pred_cv = model.predict(X_te)

        acc_scores.append(accuracy_score(y_te, y_pred_cv))
        prec_scores.append(precision_score(y_te, y_pred_cv, average="weighted", zero_division=0))
        rec_scores.append(recall_score(y_te, y_pred_cv, average="weighted", zero_division=0))
        f1_scores.append(f1_score(y_te, y_pred_cv, average="weighted", zero_division=0))

    acc = np.mean(acc_scores)
    prec = np.mean(prec_scores)
    rec = np.mean(rec_scores)
    f1 = np.mean(f1_scores)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    results.append([name, acc, prec, rec, f1])

    # Confusion Matrix
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# =====================================================
# 8) Model Karşılaştırma
# =====================================================
result_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\n===== MODEL KARŞILAŞTIRMA TABLOSU =====")
print(result_df)

melted_df = result_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Model Performans Karşılaştırması")
plt.ylim(0, 1.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

# =====================================================
# 9) Feature Importance (Random Forest)
# =====================================================
rf_model = models["Random Forest"]
rf_model.fit(X, y)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df, x="Feature", y="Importance", color="teal")
plt.title("Random Forest - En Önemli 10 Özellik")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# =====================================================
# 10) PCA (2D)
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PCA1": X_pca[:, 0],
    "PCA2": X_pca[:, 1],
    "Cover_Type": y
})

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pca_df,
    x="PCA1",
    y="PCA2",
    hue="Cover_Type",
    palette="tab10",
    s=40,
    alpha=0.7
)
plt.title("PCA: Forest Cover Type (2D Gösterim)")
plt.show()
