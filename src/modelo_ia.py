# modelo_ia.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class ModeloSemaforoIA:
    def __init__(self):
        self.modelo = LinearRegression()
        self.entrenado = False

    def entrenar(self, cantidad_muestras=50, ciclo_total=100, random_seed=42):
        np.random.seed(random_seed)
        vehiculos = np.random.randint(10, 100, cantidad_muestras)
        carriles = np.random.choice([1, 1.5, 2, 3], cantidad_muestras)
        ciclos = np.full(cantidad_muestras, ciclo_total)

        tiempo_objetivo = (vehiculos / (carriles * 2)) + np.random.normal(0, 2, cantidad_muestras)
        tiempo_objetivo = np.clip(tiempo_objetivo, 10, 60)

        self.df_entrenamiento = pd.DataFrame({
            "vehiculos": vehiculos,
            "carriles": carriles,
            "ciclo_total": ciclos,
            "tiempo_verde_objetivo": tiempo_objetivo
        })

        X = self.df_entrenamiento[["vehiculos", "carriles", "ciclo_total"]]
        y = self.df_entrenamiento["tiempo_verde_objetivo"]
        self.modelo.fit(X, y)
        self.entrenado = True

    def predecir(self, vehiculos, carriles, ciclo_total=100):
        if not self.entrenado:
            raise ValueError("El modelo aún no ha sido entrenado.")
        X_nuevo = pd.DataFrame([[vehiculos, carriles, ciclo_total]],
                               columns=["vehiculos", "carriles", "ciclo_total"])
        return self.modelo.predict(X_nuevo)[0]

    def mostrar_coeficientes(self):
        if not self.entrenado:
            print("Modelo no entrenado.")
            return
        for var, coef in zip(["vehiculos", "carriles", "ciclo_total"], self.modelo.coef_):
            print(f"{var}: {coef:.3f}")
        print(f"Intercepto: {self.modelo.intercept_:.3f}")

    def graficar_resultado(self):
        if not self.entrenado:
            raise ValueError("El modelo aún no ha sido entrenado.")
        X = self.df_entrenamiento[["vehiculos", "carriles", "ciclo_total"]]
        y_real = self.df_entrenamiento["tiempo_verde_objetivo"]
        y_pred = self.modelo.predict(X)

        plt.figure(figsize=(6, 4))
        plt.scatter(y_real, y_pred, color='blue', alpha=0.7, label="Predicción")
        plt.plot([10, 60], [10, 60], '--', color='green', label="Ideal")
        plt.xlabel("Tiempo verde real (s)")
        plt.ylabel("Tiempo verde predicho (s)")
        plt.title("Predicción del tiempo verde con IA")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
