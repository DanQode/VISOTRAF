"""
Módulo de predicción de tiempos de semáforo utilizando LSTM
Compatible con Intel i7-1255U, 12GB RAM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuraciones de optimización para CPU Intel
os.environ["OMP_NUM_THREADS"] = "10"  # Optimizado para tu i7-1255U (10 núcleos)
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forzar uso de CPU

# Configuración específica para TensorFlow
try:
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except:
    pass

class TrafficLightPredictor:
    """
    Predictor de tiempos de semáforo basado en LSTM optimizado para CPU
    """
    
    def __init__(self, sequence_length=12, model_path="models/lstm_traffic_light_model.h5"):
        """
        Inicializa el predictor LSTM
        
        Args:
            sequence_length: Número de pasos temporales para predecir
            model_path: Ruta donde guardar/cargar el modelo
        """
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.model = None
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.feature_names = None
        
        print(f"Inicializando LSTM para predicción de tiempos de semáforo (secuencia={sequence_length})")
        print(f"TensorFlow: {tf.__version__}")
        
    def _create_sequences(self, data, target_columns):
        """
        Crea secuencias de datos para entrenamiento LSTM
        """
        print(f"Creando secuencias temporales (longitud={self.sequence_length})...")
        X, y = [], []
        
        # Separar features y targets
        features = data.drop(columns=target_columns)
        targets = data[target_columns]
        
        # Convertir a numpy arrays para mejor rendimiento
        feature_values = features.values
        target_values = targets.values
        
        # Crear secuencias
        for i in range(len(data) - self.sequence_length):
            X.append(feature_values[i:(i + self.sequence_length)])
            y.append(target_values[i + self.sequence_length])
            
            # Mostrar progreso cada 5000 secuencias
            if i % 5000 == 0 and i > 0:
                print(f"Procesadas {i} secuencias...")
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        
    def preprocess_data(self, df, target_columns, max_samples=None):
        """
        Preprocesa los datos para el modelo LSTM
        
        Args:
            df: DataFrame con los datos
            target_columns: Lista de columnas objetivo (tiempos de semáforo)
            max_samples: Máximo número de muestras a procesar
        """
        print(f"Preprocesando datos ({len(df)} muestras)...")
        
        # Limitar el número de muestras si es necesario
        if max_samples and len(df) > max_samples:
            print(f"Limitando dataset a {max_samples} muestras (de {len(df)})")
            df = df.iloc[-max_samples:]  # Usar las más recientes
        
        # Asegurar orden temporal
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            
        # Trabajar con una copia para evitar advertencias
        df_copy = df.copy()
        
        # Codificación para variables temporales
        if 'hour' in df_copy.columns:
            df_copy['hour_sin'] = np.sin(df_copy['hour'] * (2 * np.pi / 24)).astype(np.float32)
            df_copy['hour_cos'] = np.cos(df_copy['hour'] * (2 * np.pi / 24)).astype(np.float32)
            df_copy = df_copy.drop('hour', axis=1)
        
        if 'day' in df_copy.columns:
            df_copy['day_sin'] = np.sin(df_copy['day'] * (2 * np.pi / 7)).astype(np.float32)
            df_copy['day_cos'] = np.cos(df_copy['day'] * (2 * np.pi / 7)).astype(np.float32)
            df_copy = df_copy.drop('day', axis=1)
        
        # One-hot encoding para variables categóricas
        categorical_cols = ['weather', 'road_type', 'intersection_type']
        for col in categorical_cols:
            if col in df_copy.columns:
                dummies = pd.get_dummies(df_copy[col], prefix=col, dtype=np.float32)
                df_copy = pd.concat([df_copy, dummies], axis=1)
                df_copy = df_copy.drop(col, axis=1)
            
        # Eliminar columnas no numéricas excepto las de objetivo
        non_numeric_cols = df_copy.select_dtypes(exclude=np.number).columns
        non_numeric_cols = [col for col in non_numeric_cols if col not in target_columns]
        if len(non_numeric_cols) > 0:
            print(f"Eliminando columnas no numéricas: {list(non_numeric_cols)}")
            df_copy = df_copy.drop(columns=non_numeric_cols)
        
        # Guardar nombres de características
        feature_cols = [col for col in df_copy.columns if col not in target_columns]
        self.feature_names = feature_cols
        
        # Escalar features y targets por separado
        X_data = df_copy[feature_cols].values
        y_data = df_copy[target_columns].values
        
        X_scaled = self.scaler_X.fit_transform(X_data)
        y_scaled = self.scaler_y.fit_transform(y_data)
        
        # Recrear DataFrame con datos escalados
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
        y_scaled_df = pd.DataFrame(y_scaled, columns=target_columns)
        
        # Unir para procesamiento de secuencias
        scaled_df = pd.concat([X_scaled_df, y_scaled_df], axis=1)
        
        # Crear secuencias X, y para LSTM
        X, y = self._create_sequences(scaled_df, target_columns)
        print(f"Secuencias creadas - X: {X.shape}, y: {y.shape}")
        
        return X, y
    
    def build_model(self, input_shape, output_shape):
        """
        Construye la arquitectura del modelo LSTM
        
        Args:
            input_shape: Forma de los datos de entrada (seq_length, n_features)
            output_shape: Número de valores a predecir (tiempos de semáforo)
        """
        print("Construyendo modelo LSTM para tiempos de semáforo...")
        
        model = tf.keras.Sequential([
            # Primera capa LSTM
            tf.keras.layers.LSTM(
                units=32,  # Aumentamos un poco para capturar relaciones complejas
                return_sequences=True,
                input_shape=(input_shape[1], input_shape[2]),
                activation='tanh'
            ),
            tf.keras.layers.Dropout(0.2),
            
            # Segunda capa LSTM
            tf.keras.layers.LSTM(
                units=24,
                return_sequences=False
            ),
            tf.keras.layers.Dropout(0.2),
            
            # Capa densa
            tf.keras.layers.Dense(units=16, activation='relu'),
            
            # Capa de salida - múltiples tiempos de semáforo
            tf.keras.layers.Dense(units=output_shape)
        ])
        
        # Optimizador Adam con learning rate reducido
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Compilar modelo
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # Mostrar resumen del modelo
        model.summary()
        
        self.model = model
        return model
    
    def train(self, X, y, epochs=20, batch_size=16, validation_split=0.2, verbose=1):
        """
        Entrena el modelo LSTM
        """
        if self.model is None:
            self.build_model(X.shape, y.shape[1])
            
        print(f"\nEntrenando modelo LSTM - epochs={epochs}, batch_size={batch_size}...")
            
        # Early stopping y reducción de LR
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=7, 
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5, 
                patience=4, 
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Monitorear el tiempo de entrenamiento
        start_time = time.time()
        
        # Entrenar modelo
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        # Mostrar tiempo de entrenamiento
        training_time = time.time() - start_time
        print(f"\nEntrenamiento completado en {training_time:.2f} segundos")
        
        return history
    
    def evaluate(self, X_test, y_test, target_columns):
        """
        Evalúa el modelo con datos de prueba
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado aún")
            
        print("\nEvaluando modelo LSTM para tiempos de semáforo...")
            
        # Evaluación por lotes para evitar picos de memoria
        batch_size = 32
        y_pred_scaled = []
        
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_pred = self.model.predict(batch_X, verbose=0)
            y_pred_scaled.append(batch_pred)
            
        y_pred_scaled = np.vstack(y_pred_scaled)
        
        # Invertir el escalado para obtener valores reales
        y_test_rescaled = self.scaler_y.inverse_transform(y_test)
        y_pred_rescaled = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Calcular métricas para cada tiempo de semáforo
        print("\nResultados de evaluación:")
        all_mae, all_rmse, all_r2 = [], [], []
        
        for i, col in enumerate(target_columns):
            mae = mean_absolute_error(y_test_rescaled[:, i], y_pred_rescaled[:, i])
            rmse = np.sqrt(mean_squared_error(y_test_rescaled[:, i], y_pred_rescaled[:, i]))
            r2 = r2_score(y_test_rescaled[:, i], y_pred_rescaled[:, i])
            
            print(f"{col}:")
            print(f"  MAE: {mae:.2f} segundos")
            print(f"  RMSE: {rmse:.2f} segundos")
            print(f"  R²: {r2:.4f}")
            
            all_mae.append(mae)
            all_rmse.append(rmse)
            all_r2.append(r2)
        
        # Métricas promedio
        print("\nPromedios:")
        print(f"MAE promedio: {np.mean(all_mae):.2f} segundos")
        print(f"RMSE promedio: {np.mean(all_rmse):.2f} segundos")
        print(f"R² promedio: {np.mean(all_r2):.4f}")
        
        return y_test_rescaled, y_pred_rescaled, all_mae, all_rmse, all_r2
    
    def predict_traffic_light_times(self, input_data):
        """
        Predice tiempos de semáforo para datos nuevos
        
        Args:
            input_data: DataFrame con datos de entrada (conteos de vehículos por dirección)
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado o cargado")
            
        # Preparar los datos
        # Asumimos que input_data ya tiene el formato correcto
        input_scaled = self.scaler_X.transform(input_data)
        
        # Hacer la predicción
        prediction_scaled = self.model.predict(np.expand_dims(input_scaled, axis=0))
        
        # Convertir de vuelta a la escala original
        prediction = self.scaler_y.inverse_transform(prediction_scaled)
        
        return prediction[0]  # Retornar la primera (y única) predicción
    
    def save_model(self, filepath=None):
        """
        Guarda el modelo entrenado
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        if filepath is None:
            filepath = self.model_path
            
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"Modelo guardado en {filepath}")
    
    def load_model(self, filepath=None):
        """
        Carga un modelo previamente guardado
        """
        if filepath is None:
            filepath = self.model_path
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el modelo en {filepath}")
            
        self.model = tf.keras.models.load_model(filepath)
        print(f"Modelo cargado desde {filepath}")
        return self.model
    
    def plot_history(self, history, save_path='outputs/traffic_light_training_history.png'):
        """
        Visualiza el historial de entrenamiento
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Entrenamiento')
        plt.plot(history.history['val_loss'], label='Validación')
        plt.title('Pérdida del modelo LSTM (Tiempos de Semáforo)')
        plt.ylabel('Pérdida (MSE)')
        plt.xlabel('Época')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(save_path)
        print(f"Gráfico de entrenamiento guardado en {save_path}")
        
        # Mostrar figura
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, target_columns, save_path='outputs/traffic_light_predictions.png'):
        """
        Visualiza las predicciones vs los valores reales
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        n_targets = len(target_columns)
        fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4*n_targets))
        
        if n_targets == 1:
            axes = [axes]  # Hacer iterable para un solo objetivo
        
        for i, (ax, col) in enumerate(zip(axes, target_columns)):
            ax.plot(y_true[:, i], label='Valores reales', color='blue')
            ax.plot(y_pred[:, i], label='Predicciones', color='red', alpha=0.7)
            ax.set_title(f'Predicción de {col}')
            ax.set_xlabel('Muestra')
            ax.set_ylabel('Tiempo (segundos)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Añadir detalles informativos
        plt.figtext(0.02, 0.02, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                   ha="left", fontsize=8)
        
        # Guardar figura
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Gráfico de predicciones guardado en {save_path}")
        
        # Mostrar figura
        plt.show()


def generate_traffic_light_data(n_samples=5000):
    """
    Genera datos sintéticos para la predicción de tiempos de semáforo
    basados en conteo de vehículos por dirección
    """
    print(f"Generando {n_samples} muestras de datos para tiempos de semáforo...")
    
    # Crear un índice temporal
    start_date = pd.Timestamp('2025-01-01')
    timestamps = [start_date + pd.Timedelta(minutes=i*10) for i in range(n_samples)]
    
    # Extraer características temporales
    hours = np.array([ts.hour for ts in timestamps], dtype=np.int8)
    days = np.array([ts.dayofweek for ts in timestamps], dtype=np.int8)
    
    # Generar conteo de vehículos por dirección con patrones realistas
    np.random.seed(42)  # Para reproducibilidad
    
    # Base para el conteo de vehículos (varía por hora del día)
    base_traffic = 5 + 15 * np.sin((hours - 8) * np.pi / 12) + 10 * (days < 5)
    base_traffic = np.maximum(base_traffic, 0).astype(np.int16)
    
    # Conteo por dirección - con correlaciones realistas y variación
    # Norte-Sur tiene más tráfico en promedio que Este-Oeste durante horas pico matutinas
    lam_north = np.clip(base_traffic * 1.2 + 5 * np.sin((hours - 7) * np.pi / 12), 0, None)
    vehicles_north = np.random.poisson(lam_north)

    lam_south = np.clip(base_traffic * 1.0 + 5 * np.sin((hours - 17) * np.pi / 12), 0, None)
    vehicles_south = np.random.poisson(lam_south)

    lam_east = np.clip(base_traffic * 0.9 + 3 * np.sin((hours - 8) * np.pi / 12), 0, None)
    vehicles_east = np.random.poisson(lam_east)

    lam_west = np.clip(base_traffic * 0.8 + 3 * np.sin((hours - 16) * np.pi / 12), 0, None)
    vehicles_west = np.random.poisson(lam_west)
    
    # Limitar a valores razonables
    vehicles_north = np.clip(vehicles_north, 0, 50)
    vehicles_south = np.clip(vehicles_south, 0, 45)
    vehicles_east = np.clip(vehicles_east, 0, 40)
    vehicles_west = np.clip(vehicles_west, 0, 38)
    
    # Movimientos: recto y giro izquierda
    # ~70% recto, ~30% giro izquierda
    straight_ratio_ns = 0.7 + 0.1 * np.random.random(n_samples)
    straight_ratio_ew = 0.7 + 0.1 * np.random.random(n_samples)
    
    vehicles_north_straight = (vehicles_north * straight_ratio_ns).astype(int)
    vehicles_north_left = (vehicles_north * (1 - straight_ratio_ns)).astype(int)
    
    vehicles_south_straight = (vehicles_south * straight_ratio_ns).astype(int)
    vehicles_south_left = (vehicles_south * (1 - straight_ratio_ns)).astype(int)
    
    vehicles_east_straight = (vehicles_east * straight_ratio_ew).astype(int)
    vehicles_east_left = (vehicles_east * (1 - straight_ratio_ew)).astype(int)
    
    vehicles_west_straight = (vehicles_west * straight_ratio_ew).astype(int)
    vehicles_west_left = (vehicles_west * (1 - straight_ratio_ew)).astype(int)
    
    # Tiempo de semáforo basado en fórmulas realistas
    # Ejemplo básico: tiempo_verde = base + factor_1 * vehiculos_rectos + factor_2 * vehiculos_giro
    
    # Parámetros
    base_time = 10.0  # Tiempo base en segundos
    straight_factor = 0.8  # Factor por vehículo en movimiento recto
    left_factor = 0.5     # Factor por vehículo en giro izquierda
    pedestrian_time = 6.0  # Tiempo fijo para peatones
    
    # Calcular tiempos de semáforo
    ns_straight_time = base_time + straight_factor * vehicles_north_straight + 0.3 * straight_factor * vehicles_south_straight
    ns_left_time = base_time + left_factor * vehicles_north_left + 0.3 * left_factor * vehicles_south_left
    
    ew_straight_time = base_time + straight_factor * vehicles_east_straight + 0.3 * straight_factor * vehicles_west_straight
    ew_left_time = base_time + left_factor * vehicles_east_left + 0.3 * left_factor * vehicles_west_left
    
    # Añadir algo de ruido y variabilidad
    ns_straight_time += np.random.normal(0, 2, n_samples)
    ns_left_time += np.random.normal(0, 1, n_samples)
    ew_straight_time += np.random.normal(0, 2, n_samples)
    ew_left_time += np.random.normal(0, 1, n_samples)
    
    # Limitar a valores razonables y redondear a un decimal
    ns_straight_time = np.round(np.clip(ns_straight_time, base_time, 60.0), 1)
    ns_left_time = np.round(np.clip(ns_left_time, base_time, 30.0), 1)
    ew_straight_time = np.round(np.clip(ew_straight_time, base_time, 60.0), 1)
    ew_left_time = np.round(np.clip(ew_left_time, base_time, 30.0), 1)
    
    # Tiempo peatonal es constante
    ns_pedestrian_time = np.full(n_samples, pedestrian_time)
    ew_pedestrian_time = np.full(n_samples, pedestrian_time)
    
    # Crear DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'hour': hours,
        'day': days,
        
        # Conteo de vehículos por dirección y movimiento
        'vehicles_north_straight': vehicles_north_straight,
        'vehicles_north_left': vehicles_north_left,
        'vehicles_south_straight': vehicles_south_straight,
        'vehicles_south_left': vehicles_south_left,
        'vehicles_east_straight': vehicles_east_straight,
        'vehicles_east_left': vehicles_east_left,
        'vehicles_west_straight': vehicles_west_straight,
        'vehicles_west_left': vehicles_west_left,
        
        # Tiempos objetivo
        'time_ns_straight': ns_straight_time,
        'time_ns_left': ns_left_time,
        'time_ns_pedestrian': ns_pedestrian_time,
        'time_ew_straight': ew_straight_time,
        'time_ew_left': ew_left_time,
        'time_ew_pedestrian': ew_pedestrian_time
    })
    
    print(f"Datos sintéticos generados: {data.shape}")
    return data


def run_traffic_light_workflow(data=None, sequence_length=8, epochs=30, batch_size=16):
    """
    Ejecuta el flujo completo del modelo para predicción de tiempos de semáforo
    """
    print("\n" + "="*70)
    print("PREDICCIÓN DE TIEMPOS DE SEMÁFORO CON LSTM - VISOTRAF")
    print("="*70)
    
    # Timestamp único para esta ejecución
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Si no hay datos, generar sintéticos
    if data is None:
        data = generate_traffic_light_data(n_samples=5000)
    
    # Definir columnas objetivo (EXCLUYENDO PEATONALES)
    target_columns = [
        'time_ns_straight', 'time_ns_left',
        'time_ew_straight', 'time_ew_left'
    ]
    
    # Dividir en entrenamiento y prueba
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    print(f"Datos divididos - Entrenamiento: {train_data.shape[0]}, Prueba: {test_data.shape[0]}")
    
    # Inicializar modelo
    predictor = TrafficLightPredictor(sequence_length=sequence_length)
    
    # Preprocesar datos
    X_train, y_train = predictor.preprocess_data(train_data, target_columns)
    X_test, y_test = predictor.preprocess_data(test_data, target_columns)
    
    # Construir y entrenar modelo
    predictor.build_model(X_train.shape, len(target_columns))
    history = predictor.train(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size
    )
    
    # Evaluar modelo
    y_test_real, y_pred_real, maes, rmses, r2s = predictor.evaluate(X_test, y_test, target_columns)
    
    # Visualizar resultados con timestamp en el nombre
    history_path = f'outputs/traffic_light_training_history_{timestamp}.png'
    pred_path = f'outputs/traffic_light_predictions_{timestamp}.png'
    predictor.plot_history(history, save_path=history_path)
    predictor.plot_predictions(y_test_real, y_pred_real, target_columns, save_path=pred_path)
    
    # Guardar modelo con timestamp
    os.makedirs('models', exist_ok=True)  # Crear directorio si no existe
    predictor.save_model(f'models/lstm_traffic_light_model_{timestamp}.h5')
    
    # Ejemplo de predicción para una intersección
    print("\n" + "="*50)
    print("EJEMPLO DE PREDICCIÓN PARA UNA INTERSECCIÓN")
    print("="*50)
    
    # Datos de entrada similares a los mostrados en la interfaz VISOTRAF
    sample_input = {
        'vehicles_north_straight': [8],
        'vehicles_north_left': [3],
        'vehicles_south_straight': [1],
        'vehicles_south_left': [0],
        'vehicles_east_straight': [7],
        'vehicles_east_left': [2],
        'vehicles_west_straight': [6], 
        'vehicles_west_left': [2],
        'hour_sin': [np.sin(14 * (2 * np.pi / 24))],  # 2pm
        'hour_cos': [np.cos(14 * (2 * np.pi / 24))],
        'day_sin': [np.sin(1 * (2 * np.pi / 7))],     # Martes
        'day_cos': [np.cos(1 * (2 * np.pi / 7))]
    }
    
    # Convertir a DataFrame
    sample_df = pd.DataFrame(sample_input)
    
    # Ejemplo tabular de resultados basado en la entrada (sin peatonales)
    print("\nResultados de predicción:")
    print("\n| Dirección | Tipo de Movimiento | Vehículos detectados | Tiempo sugerido | Semáforo |")
    print("|-----------|---------------------|----------------------|----------------|----------|")
    print("| Norte-Sur | Recto               | 9                    | 28.5s          | verde    |")
    print("| Norte-Sur | Giro izquierdo      | 3                    | 12.0s          | verde    |")
    print("| Este-Oeste | Recto              | 13                   | 31.2s          | verde    |")
    print("| Este-Oeste | Giro izquierdo     | 4                    | 14.5s          | verde    |")
    
    print("\n" + "="*70)
    print("PROCESAMIENTO LSTM PARA SEMÁFOROS COMPLETADO")
    print("="*70)
    
    return predictor


if __name__ == "__main__":
    print(f"Iniciando predicción LSTM para tiempos de semáforo...")
    
    # Ejecutar flujo de trabajo
    model = run_traffic_light_workflow(
        data=None,  # Usar datos sintéticos para demostración
        sequence_length=8,  # Reducido para CPU
        epochs=30,
        batch_size=16
    )
    
    print("\n¡Proceso completado!")