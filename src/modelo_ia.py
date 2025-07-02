from sklearn.linear_model import LinearRegression
import pandas as pd

def entrenar_modelo(df):
    X = df[['vehiculos_detectados']]
    y = df['tiempo_verde']  # Este campo debe estar en tu CSV de entrenamiento
    modelo = LinearRegression()
    modelo.fit(X, y)
    return modelo

def predecir(modelo, vehiculos):
    return modelo.predict([[vehiculos]])[0]
