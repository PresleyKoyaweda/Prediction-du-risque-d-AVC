import os
import mlflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, recall_score,
    precision_score, f1_score, confusion_matrix
)
from mlflow.models.signature import infer_signature
from joblib import dump

# === Chargement des données ===
df_trainset = pd.read_csv("D:/Perso/Detection_AVC/data/train.csv")
df_testset = pd.read_csv("D:/Perso/Detection_AVC/data/test.csv")

numeric_vars = ["age", "avg_glucose_level", "bmi"]
categorical_vars = [
    'gender', 'ever_married', 'work_type',
    'Residence_type', 'smoking_status',
    'hypertension', 'heart_disease'
]

X_train = df_trainset.drop("stroke", axis=1)
y_train = df_trainset["stroke"]
X_test = df_testset.drop("stroke", axis=1)
y_test = df_testset["stroke"]

# === Pipelines ===
numeric_pipeline = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_vars),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_vars)
])

model_pipeline = ImbPipeline([
    ("preprocessing", preprocessor),
    ("resample", SMOTE(random_state=42)),
    ("poly", PolynomialFeatures(degree=2)),
    ("var_filter", VarianceThreshold(threshold=0.0)),
    ("select", SelectKBest(score_func=f_classif, k=30)),
    ("logreg", LogisticRegression(solver="saga", max_iter=500, C=1))
])

# === Enregistrement avec MLflow ===
mlflow.set_experiment("avc_detection_final")
mlflow.end_run()

name = "LogisticRegression_Optimised"
input_example = X_test[:1].to_dict(orient="records")[0]

with mlflow.start_run(run_name=name):
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

    # Log metrics & params
    mlflow.log_param("model", name)
    mlflow.log_metrics(metrics)

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matrice de confusion")
    plt.xlabel("Prédiction")
    plt.ylabel("Réalité")
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    os.remove(cm_path)

    # Log modèle
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        sk_model=model_pipeline,
        artifact_path="model",
        input_example=input_example,
        signature=signature
    )

    dump(model_pipeline, "D:/Perso/Detection_AVC/models/final_model.joblib") 
    #dump(model_pipeline, os.path.join("models", "final_model.joblib"))

    print("✅ Entraînement terminé et modèle sauvegardé.")
