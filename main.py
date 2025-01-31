from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib  # Para guardar y cargar el modelo
from pathlib import Path

# Configuración de FastAPI
app = FastAPI()

# 1. Crear un conjunto de datos de ejemplo
def crear_datos():
    np.random.seed(42)  # Para reproducibilidad
    n_samples = 200  # Número de muestras

    # Generar datos sintéticos
    calificaciones = np.random.uniform(0, 10, n_samples)  # Calificaciones entre 0 y 10
    dificultad = np.random.randint(1, 4, n_samples)       # Dificultad: 1, 2 o 3
    tiempo = np.random.uniform(0, 10, n_samples)          # Tiempo entre 0 y 10 minutos

    # Regla simple para aprobar: calificación >= 6
    aprobado = np.where(calificaciones >= 6, 1, 0)

    # Crear un DataFrame
    data = pd.DataFrame({
        'Calificacion': calificaciones,
        'Dificultad': dificultad,
        'Tiempo': tiempo,
        'Aprobado': aprobado
    })
    return data

# 2. Entrenar el modelo (o cargarlo si ya existe)
def entrenar_modelo():
    modelo_path = Path("modelo_regresion_logistica.pkl")
    if modelo_path.exists():
        # Cargar el modelo si ya existe
        modelo = joblib.load(modelo_path)
    else:
        # Crear y entrenar el modelo
        data = crear_datos()
        X = data[['Calificacion', 'Dificultad', 'Tiempo']]  # Características
        y = data['Aprobado']                                # Etiqueta

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = LogisticRegression()
        modelo.fit(X_train, y_train)

        # Guardar el modelo para uso futuro
        joblib.dump(modelo, modelo_path)
    return modelo

# Cargar el modelo al iniciar la API
modelo = entrenar_modelo()

# 3. Definir el esquema de entrada para la API
class PrediccionRequest(BaseModel):
    calificacion: float
    dificultad: int
    tiempo: float

# 4. Crear el endpoint de predicción
@app.post("/predecir")
def predecir_aprobacion(datos: PrediccionRequest):
    try:
        # Convertir los datos de entrada en un array de numpy
        entrada = np.array([[datos.calificacion, datos.dificultad, datos.tiempo]])

        # Hacer la predicción
        prediccion = modelo.predict(entrada)

        # Devolver el resultado
        return {"aprobado": bool(prediccion[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 5. Iniciar la API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)