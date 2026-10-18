import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Configurar el experimento en MLflow
mlflow.set_experiment("MLflow Quickstart - Iris Classification")

# Cargar el dataset
X, y = datasets.load_iris(return_X_y=True)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiperparámetros
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

mlflow.sklearn.autolog()

# Entrenar el modelo
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
