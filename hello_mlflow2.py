import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import mlflow
import mlflow.sklearn

# Configurar URI de almacenamiento local
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Definir el experimento
mlflow.set_experiment("Hello MLFlow con Gráficas")

# Cargar datos
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir hiperparámetros
params = {
    "n_estimators": 30,
    "max_depth": 5,
    "random_state": 42
}

# Ejecución en MLflow
with mlflow.start_run(run_name="Prueba_RandomForest"):
    
    # Entrenar el modelo
    rf_model = RandomForestClassifier(**params)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    # Calcular Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    
    # Registrar Parámetros y Métricas (Se usa log_params para el diccionario)
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    
    # Crear y guardar Artefacto (Matriz de Confusión)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor Real")
    ax.set_title("Matriz de Confusión - Random Forest")
    
    plot_path = "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    
    # Enviar la imagen como artefacto a MLflow
    mlflow.log_artifact(plot_path)
    
    # Registrar el modelo entrenado en MLflow
    mlflow.sklearn.log_model(
    rf_model, 
    artifact_path="model", 
    skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    
    # Limpieza local de la imagen temporal
    if os.path.exists(plot_path):
        os.remove(plot_path)

print(f"Ejecución con Accuracy: {acc:.4f}")