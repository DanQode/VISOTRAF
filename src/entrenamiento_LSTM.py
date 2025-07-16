"""
Módulo de predicción de tiempos de semáforo utilizando LSTM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import os
import time
import pickle
import json
import shutil
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configuraciones de optimización para CPU Intel
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except:
    pass

FEATURE_NAMES = [
    'vehicles_north_straight', 'vehicles_north_left',
    'vehicles_south_straight', 'vehicles_south_left',
    'vehicles_east_straight', 'vehicles_east_left',
    'vehicles_west_straight', 'vehicles_west_left',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

TARGET_COLUMNS = [
    'time_ns_straight', 'time_ns_left', 'time_ew_straight', 'time_ew_left'
]

class TrafficLightPredictor:
    """
    Predictor de tiempos de semáforo basado en LSTM optimizado para CPU
    SOLO usa las 12 columnas de features esperadas por predicción_LSTM
    """
    def __init__(self, sequence_length=8, model_path="models/lstm_traffic_light_model.h5"):
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.model = None
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.feature_names = FEATURE_NAMES
        self.target_columns = TARGET_COLUMNS

        print(f"Inicializando LSTM para predicción de tiempos de semáforo (secuencia={sequence_length})")
        print(f"TensorFlow: {tf.__version__}")

    def _create_sequences(self, data, target_columns):
        X, y = [], []
        features = data[self.feature_names]
        targets = data[target_columns]
        feature_values = features.values
        target_values = targets.values
        for i in range(len(data) - self.sequence_length):
            X.append(feature_values[i:(i + self.sequence_length)])
            y.append(target_values[i + self.sequence_length])
            if i % 5000 == 0 and i > 0:
                print(f"Procesadas {i} secuencias...")
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def preprocess_data(self, df, target_columns, max_samples=None):
        print(f"Preprocesando datos ({len(df)} muestras)...")

        if max_samples and len(df) > max_samples:
            print(f"Limitando dataset a {max_samples} muestras (de {len(df)})")
            df = df.iloc[-max_samples:]

        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24)).astype(np.float32)
            df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24)).astype(np.float32)
        if 'day' in df.columns:
            df['day_sin'] = np.sin(df['day'] * (2 * np.pi / 7)).astype(np.float32)
            df['day_cos'] = np.cos(df['day'] * (2 * np.pi / 7)).astype(np.float32)

        for col in self.feature_names:
            if col not in df.columns:
                raise ValueError(f"Falta columna requerida: {col}")

        X_data = df[self.feature_names].values
        y_data = df[target_columns].values

        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        X_scaled = self.scaler_X.fit_transform(X_data)
        y_scaled = self.scaler_y.fit_transform(y_data)

        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        y_scaled_df = pd.DataFrame(y_scaled, columns=target_columns)
        scaled_df = pd.concat([X_scaled_df, y_scaled_df], axis=1)

        X, y = self._create_sequences(scaled_df, target_columns)
        print(f"Secuencias creadas - X: {X.shape}, y: {y.shape}")
        return X, y

    def build_model(self, input_shape, output_shape):
        print("Construyendo modelo LSTM para tiempos de semáforo...")
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                units=32,
                return_sequences=True,
                input_shape=(input_shape[1], input_shape[2]),
                activation='tanh'
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(
                units=24,
                return_sequences=False
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=output_shape)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        model.summary()
        self.model = model
        return model

    def train(self, X, y, epochs=20, batch_size=16, validation_split=0.2, verbose=1):
        if self.model is None:
            self.build_model(X.shape, y.shape[1])
        print(f"\nEntrenando modelo LSTM - epochs={epochs}, batch_size={batch_size}...")
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
        start_time = time.time()
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        training_time = time.time() - start_time
        print(f"\nEntrenamiento completado en {training_time:.2f} segundos")
        return history

    def evaluate(self, X_test, y_test, target_columns):
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado aún")
        print("\nEvaluando modelo LSTM para tiempos de semáforo...")
        batch_size = 32
        y_pred_scaled = []
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_pred = self.model.predict(batch_X, verbose=0)
            y_pred_scaled.append(batch_pred)
        y_pred_scaled = np.vstack(y_pred_scaled)
        y_test_rescaled = self.scaler_y.inverse_transform(y_test)
        y_pred_rescaled = self.scaler_y.inverse_transform(y_pred_scaled)
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
        print("\nPromedios:")
        print(f"MAE promedio: {np.mean(all_mae):.2f} segundos")
        print(f"RMSE promedio: {np.mean(all_rmse):.2f} segundos")
        print(f"R² promedio: {np.mean(all_r2):.4f}")
        return y_test_rescaled, y_pred_rescaled, all_mae, all_rmse, all_r2

    def predict_traffic_light_times(self, input_data):
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado o cargado")
        for col in self.feature_names:
            if col not in input_data.columns:
                raise ValueError(f"Falta columna requerida: {col}")
        input_scaled = self.scaler_X.transform(input_data[self.feature_names])
        prediction_scaled = self.model.predict(np.expand_dims(input_scaled, axis=0))
        prediction = self.scaler_y.inverse_transform(prediction_scaled)
        return prediction[0]

    def save_model(self, filepath=None):
        """
        Guarda el modelo entrenado y los scalers en un .pkl
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        if filepath is None:
            filepath = self.model_path
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        scaler_path = filepath.replace('.h5', '_scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'feature_names': self.feature_names,
                'target_columns': self.target_columns
            }, f)
        print(f"Modelo guardado en {filepath}")
        print(f"Scalers guardados en {scaler_path}")

    def load_model(self, filepath=None):
        """
        Carga modelo Keras y los scalers desde el .pkl correspondiente
        """
        if filepath is None:
            filepath = self.model_path
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el modelo en {filepath}")
        self.model = tf.keras.models.load_model(filepath)
        scaler_path = filepath.replace('.h5', '_scalers.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers_data = pickle.load(f)
                self.scaler_X = scalers_data['scaler_X']
                self.scaler_y = scalers_data['scaler_y']
                self.feature_names = scalers_data['feature_names']
                self.target_columns = scalers_data.get('target_columns')
            print(f"Modelo y scalers cargados desde {filepath}")
        else:
            print(f"Modelo cargado desde {filepath} (sin scalers)")
        return self.model

    def plot_history(self, history, save_path='outputs/traffic_light_training_history.png'):
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
        plt.savefig(save_path)
        print(f"Gráfico de entrenamiento guardado en {save_path}")
        plt.show()

    def plot_predictions(self, y_true, y_pred, target_columns, save_path='outputs/traffic_light_predictions.png'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        n_targets = len(target_columns)
        fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4*n_targets))
        if n_targets == 1:
            axes = [axes]
        for i, (ax, col) in enumerate(zip(axes, target_columns)):
            ax.plot(y_true[:, i], label='Valores reales', color='blue')
            ax.plot(y_pred[:, i], label='Predicciones', color='red', alpha=0.7)
            ax.set_title(f'Predicción de {col}')
            ax.set_xlabel('Muestra')
            ax.set_ylabel('Tiempo (segundos)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.figtext(0.02, 0.02, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                   ha="left", fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Gráfico de predicciones guardado en {save_path}")
        plt.show()


def export_model_for_production(predictor, filepath='models/production_model.h5'):
    print(f"Exportando modelo para producción en {filepath}...")
    predictor.save_model(filepath)
    metadata = {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sequence_length': predictor.sequence_length,
        'feature_names': predictor.feature_names,
        'target_columns': predictor.target_columns,
        'tensorflow_version': tf.__version__
    }
    meta_path = filepath.replace('.h5', '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Modelo exportado correctamente con metadatos en {meta_path}")
    print("El modelo está listo para ser usado con prediccion_LSTM.py")
    return filepath

def generate_traffic_light_data(n_samples=5000):
    print(f"Generando {n_samples} muestras de datos para tiempos de semáforo...")
    start_date = pd.Timestamp('2025-01-01')
    timestamps = [start_date + pd.Timedelta(minutes=i*10) for i in range(n_samples)]
    hours = np.array([ts.hour for ts in timestamps], dtype=np.int8)
    days = np.array([ts.dayofweek for ts in timestamps], dtype=np.int8)

    np.random.seed(42)
    base_traffic = 5 + 15 * np.sin((hours - 8) * np.pi / 12) + 10 * (days < 5)
    base_traffic = np.maximum(base_traffic, 0).astype(np.int16)

    lam_north = np.clip(base_traffic * 1.2 + 5 * np.sin((hours - 7) * np.pi / 12), 0, None)
    vehicles_north = np.random.poisson(lam_north)
    lam_south = np.clip(base_traffic * 1.0 + 5 * np.sin((hours - 17) * np.pi / 12), 0, None)
    vehicles_south = np.random.poisson(lam_south)
    lam_east = np.clip(base_traffic * 0.9 + 3 * np.sin((hours - 8) * np.pi / 12), 0, None)
    vehicles_east = np.random.poisson(lam_east)
    lam_west = np.clip(base_traffic * 0.8 + 3 * np.sin((hours - 16) * np.pi / 12), 0, None)
    vehicles_west = np.random.poisson(lam_west)

    vehicles_north = np.clip(vehicles_north, 0, 50)
    vehicles_south = np.clip(vehicles_south, 0, 45)
    vehicles_east = np.clip(vehicles_east, 0, 40)
    vehicles_west = np.clip(vehicles_west, 0, 38)

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

    base_time = 10.0
    straight_factor = 0.8
    left_factor = 0.5
    pedestrian_time = 6.0

    ns_straight_time = base_time + straight_factor * vehicles_north_straight + 0.3 * straight_factor * vehicles_south_straight
    ns_left_time = base_time + left_factor * vehicles_north_left + 0.3 * left_factor * vehicles_south_left
    ew_straight_time = base_time + straight_factor * vehicles_east_straight + 0.3 * straight_factor * vehicles_west_straight
    ew_left_time = base_time + left_factor * vehicles_east_left + 0.3 * left_factor * vehicles_west_left

    ns_straight_time += np.random.normal(0, 2, n_samples)
    ns_left_time += np.random.normal(0, 1, n_samples)
    ew_straight_time += np.random.normal(0, 2, n_samples)
    ew_left_time += np.random.normal(0, 1, n_samples)

    ns_straight_time = np.round(np.clip(ns_straight_time, base_time, 60.0), 1)
    ns_left_time = np.round(np.clip(ns_left_time, base_time, 30.0), 1)
    ew_straight_time = np.round(np.clip(ew_straight_time, base_time, 60.0), 1)
    ew_left_time = np.round(np.clip(ew_left_time, base_time, 30.0), 1)

    data = pd.DataFrame({
        'vehicles_north_straight': vehicles_north_straight,
        'vehicles_north_left': vehicles_north_left,
        'vehicles_south_straight': vehicles_south_straight,
        'vehicles_south_left': vehicles_south_left,
        'vehicles_east_straight': vehicles_east_straight,
        'vehicles_east_left': vehicles_east_left,
        'vehicles_west_straight': vehicles_west_straight,
        'vehicles_west_left': vehicles_west_left,
        'hour_sin': np.sin(hours * (2 * np.pi / 24)),
        'hour_cos': np.cos(hours * (2 * np.pi / 24)),
        'day_sin': np.sin(days * (2 * np.pi / 7)),
        'day_cos': np.cos(days * (2 * np.pi / 7)),
        'time_ns_straight': ns_straight_time,
        'time_ns_left': ns_left_time,
        'time_ew_straight': ew_straight_time,
        'time_ew_left': ew_left_time,
    })
    print(f"Datos sintéticos generados: {data.shape}")
    return data

def run_traffic_light_workflow(data=None, sequence_length=8, epochs=30, batch_size=16):
    print("\n" + "="*70)
    print("PREDICCIÓN DE TIEMPOS DE SEMÁFORO CON LSTM - VISOTRAF")
    print("="*70)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if data is None:
        data = generate_traffic_light_data(n_samples=5000)
    target_columns = TARGET_COLUMNS
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    print(f"Datos divididos - Entrenamiento: {train_data.shape[0]}, Prueba: {test_data.shape[0]}")
    predictor = TrafficLightPredictor(sequence_length=sequence_length)
    X_train, y_train = predictor.preprocess_data(train_data, target_columns)
    X_test, y_test = predictor.preprocess_data(test_data, target_columns)
    predictor.build_model(X_train.shape, len(target_columns))
    history = predictor.train(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size
    )
    y_test_real, y_pred_real, maes, rmses, r2s = predictor.evaluate(X_test, y_test, target_columns)
    history_path = f'outputs/traffic_light_training_history_{timestamp}.png'
    pred_path = f'outputs/traffic_light_predictions_{timestamp}.png'
    predictor.plot_history(history, save_path=history_path)
    predictor.plot_predictions(y_test_real, y_pred_real, target_columns, save_path=pred_path)
    os.makedirs('models', exist_ok=True)
    model_path = f'models/lstm_traffic_light_model_{timestamp}.h5'
    predictor.save_model(model_path)
    production_model_path = export_model_for_production(predictor, f'models/production_model_{timestamp}.h5')
    latest_path = 'models/latest_production_model.h5'
    if os.path.exists(latest_path):
        os.remove(latest_path)
    shutil.copy(production_model_path, latest_path)
    for ext in ['_scalers.pkl', '_metadata.json']:
        src = production_model_path.replace('.h5', ext)
        dst = latest_path.replace('.h5', ext)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copy(src, dst)
    print(f"Se ha creado una copia con nombre fijo en {latest_path} para facilitar la integración")
    print("\n" + "="*50)
    print("EJEMPLO DE PREDICCIÓN PARA UNA INTERSECCIÓN")
    print("="*50)
    sample_input = {
        'vehicles_north_straight': [8],
        'vehicles_north_left': [3],
        'vehicles_south_straight': [1],
        'vehicles_south_left': [0],
        'vehicles_east_straight': [7],
        'vehicles_east_left': [2],
        'vehicles_west_straight': [6],
        'vehicles_west_left': [2],
        'hour_sin': [np.sin(14 * (2 * np.pi / 24))],
        'hour_cos': [np.cos(14 * (2 * np.pi / 24))],
        'day_sin': [np.sin(1 * (2 * np.pi / 7))],
        'day_cos': [np.cos(1 * (2 * np.pi / 7))]
    }
    sample_df = pd.DataFrame(sample_input)
    y_pred = predictor.predict_traffic_light_times(sample_df)
    print("\nResultados de predicción para entrada ejemplo:")
    for i, col in enumerate(target_columns):
        print(f"{col}: {y_pred[i]:.2f} segundos")
    print("\n" + "="*70)
    print("PROCESAMIENTO LSTM PARA SEMÁFOROS COMPLETADO")
    print("="*70)
    print(f"Modelo para producción guardado en: {production_model_path}")
    print(f"Copia del modelo con nombre fijo en: {latest_path}")
    return predictor

if __name__ == "__main__":
    print(f"Iniciando predicción LSTM para tiempos de semáforo...")
    model = run_traffic_light_workflow(
        data=None,
        sequence_length=8,
        epochs=30,
        batch_size=16
    )
    print("\n¡Proceso completado!")