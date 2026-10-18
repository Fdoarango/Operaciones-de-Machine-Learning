import os
import mlflow

# Configurar URI de almacenamiento local y el nombre del experimento
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Hello MLflow Experiment")

# Crear un archivo de texto para artefacto
artifact_filename = "notes.txt"
with open(artifact_filename, "w") as f:
    f.write("Este es un archivo de notas generado durante el experimento.")

# Ejecutar mlflow
with mlflow.start_run():
    # Registrar un parámetro
    mlflow.log_param("learning_rate", 0.01)
    
    # Registrar una métrica
    mlflow.log_metric("accuracy", 0.95)
    
    # Registrar el archivo como artefacto
    mlflow.log_artifact(artifact_filename)

# Limpieza opcional del archivo local temporal
if os.path.exists(artifact_filename):
    os.remove(artifact_filename)

print("Experimento en MLflow")