from src.modelo_ia import ModeloSemaforoIA

modelo = ModeloSemaforoIA()
modelo.entrenar()  # Entrenamiento con datos simulados
modelo.mostrar_coeficientes()  # Mostrar fórmula del modelo
modelo.graficar_resultado()    # Ver predicción vs realidad

# Prueba con valores de ejemplo
tiempo = modelo.predecir(vehiculos=50, carriles=2)
print(f"Tiempo verde recomendado: {tiempo:.2f} segundos")
