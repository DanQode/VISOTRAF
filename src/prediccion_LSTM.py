"""
Adaptador para predicción de tiempos de semáforo usando LSTM con estabilización
"""

import numpy as np
import os
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import time

class LSTMTrafficPredictor:
    """
    Clase adaptadora para integrar la predicción LSTM con el dashboard
    """
    def __init__(self, model_path=None, sequence_length=8):
        """
        Inicializa el predictor de tiempos de semáforo
        
        Args:
            model_path: Ruta al modelo LSTM entrenado
            sequence_length: Longitud de secuencia que espera el modelo LSTM
        """
        self.model = None
        self.model_path = model_path or "models/lstm_traffic_light_model20250713_045748.h5"
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = sequence_length
        
        # Historial de datos para mantener secuencias
        self.data_history = []
        
        # Para manejar actualizaciones frecuentes
        self.last_prediction = None
        self.prediction_time = None
        self.last_update_time = time.time()
        self.is_processing = False
        
        # Historial de predicciones para suavizado
        self.prediction_history = []
        self.history_max_length = 5  # Mantener últimas 5 predicciones para suavizado
        
        # Cargar modelo o crear uno simple para demostración
        self.load_or_create_model()
        
        # Definimos feature_names basados en el dataset de entrenamiento
        self.feature_names = [
            'vehicles_north_straight', 'vehicles_north_left',
            'vehicles_south_straight', 'vehicles_south_left',
            'vehicles_east_straight', 'vehicles_east_left',
            'vehicles_west_straight', 'vehicles_west_left',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        print(f"Inicializado LSTMTrafficPredictor (sequence_length={sequence_length})")
    
    def load_or_create_model(self):
        """
        Carga el modelo si existe, o crea uno simple para demostración
        """
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                print(f"Modelo LSTM cargado desde: {self.model_path}")
            else:
                print(f"Modelo no encontrado en {self.model_path}, usando algoritmo de fallback")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            print("Usando algoritmo de fallback")
    
    def equalize_dashboard_data(self, vehicle_counts):
        """
        Transforma los datos del dashboard al formato esperado por el modelo LSTM
        
        Args:
            vehicle_counts: Diccionario con conteo por dirección {"Norte": n, "Sur": s, "Este": e, "Oeste": w}
        
        Returns:
            dict: Datos en formato compatible con el modelo LSTM
        """
        # Extraer conteos básicos
        north = vehicle_counts.get("Norte", 0)
        south = vehicle_counts.get("Sur", 0)
        east = vehicle_counts.get("Este", 0)
        west = vehicle_counts.get("Oeste", 0)
        
        # Obtener información temporal actual
        now = datetime.now()
        hour = now.hour
        day = now.weekday()  # 0=Lunes, 6=Domingo
        
        # Transformaciones cíclicas de hora y día
        hour_sin = np.sin(hour * (2 * np.pi / 24))
        hour_cos = np.cos(hour * (2 * np.pi / 24))
        day_sin = np.sin(day * (2 * np.pi / 7))
        day_cos = np.cos(day * (2 * np.pi / 7))
        
        # Distribuir conteos en movimientos (straight y left)
        # Asumimos que 70% de vehículos van recto y 30% giran a la izquierda
        straight_ratio = 0.7
        left_ratio = 0.3
        
        # Crear diccionario con datos equalizados
        equalized_data = {
            'vehicles_north_straight': int(north * straight_ratio),
            'vehicles_north_left': int(north * left_ratio),
            'vehicles_south_straight': int(south * straight_ratio),
            'vehicles_south_left': int(south * left_ratio),
            'vehicles_east_straight': int(east * straight_ratio),
            'vehicles_east_left': int(east * left_ratio),
            'vehicles_west_straight': int(west * straight_ratio),
            'vehicles_west_left': int(west * left_ratio),
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_sin': day_sin,
            'day_cos': day_cos
        }
        
        return equalized_data
    
    def create_sequence(self, new_data):
        """
        Mantiene un historial de datos y crea una secuencia de la longitud requerida
        
        Args:
            new_data: Diccionario con nuevos datos
            
        Returns:
            np.array: Secuencia de datos preparada para el modelo LSTM
        """
        # Añadir nuevos datos al historial
        self.data_history.append(new_data)
        
        # Mantener solo los últimos "sequence_length" registros
        if len(self.data_history) > self.sequence_length:
            self.data_history = self.data_history[-self.sequence_length:]
        
        # Si no tenemos suficientes datos, duplicamos el último hasta completar
        while len(self.data_history) < self.sequence_length:
            self.data_history.append(new_data)
        
        # Convertir a DataFrame
        sequence_df = pd.DataFrame(self.data_history)
        
        # Asegurar que tenemos todas las columnas necesarias
        for feature in self.feature_names:
            if feature not in sequence_df.columns:
                sequence_df[feature] = 0
        
        # Mantener solo las columnas que el modelo espera
        sequence_df = sequence_df[self.feature_names]
        
        # Convertir a numpy array
        sequence_array = sequence_df.values
        
        # Escalar datos si el modelo está disponible
        if hasattr(self, 'scaler_X') and self.scaler_X is not None:
            try:
                # Intentar usar el scaler previamente ajustado
                sequence_array = self.scaler_X.transform(sequence_array)
            except:
                # Si es la primera vez, ajustar el scaler
                sequence_array = self.scaler_X.fit_transform(sequence_array)
        
        # Reshape para LSTM: [1, sequence_length, n_features]
        return np.expand_dims(sequence_array, axis=0)
    
    def smooth_predictions(self, new_prediction):
        """
        Aplica suavizado exponencial a las predicciones para evitar cambios bruscos
        
        Args:
            new_prediction: Nueva predicción de tiempos
            
        Returns:
            dict: Predicción suavizada
        """
        # Añadir nueva predicción al historial
        self.prediction_history.append(new_prediction)
        
        # Mantener solo las últimas N predicciones
        if len(self.prediction_history) > self.history_max_length:
            self.prediction_history = self.prediction_history[-self.history_max_length:]
        
        # Si solo tenemos una predicción, devolverla tal cual
        if len(self.prediction_history) == 1:
            return new_prediction
        
        # Crear predicción suavizada con pesos exponenciales
        # Las predicciones más recientes tienen más peso
        smoothed_prediction = {
            'main': {'ns': 0, 'eo': 0},
            'turn': {'ns': 0, 'eo': 0}
        }
        
        total_weight = 0
        
        for i, pred in enumerate(self.prediction_history):
            # Peso exponencial: las más recientes tienen más peso
            weight = 2 ** i
            total_weight += weight
            
            # Acumular valores ponderados
            smoothed_prediction['main']['ns'] += pred['main']['ns'] * weight
            smoothed_prediction['main']['eo'] += pred['main']['eo'] * weight
            smoothed_prediction['turn']['ns'] += pred['turn']['ns'] * weight
            smoothed_prediction['turn']['eo'] += pred['turn']['eo'] * weight
        
        # Normalizar por el peso total
        smoothed_prediction['main']['ns'] /= total_weight
        smoothed_prediction['main']['eo'] /= total_weight
        smoothed_prediction['turn']['ns'] /= total_weight
        smoothed_prediction['turn']['eo'] /= total_weight
        
        # Redondear a un decimal
        smoothed_prediction['main']['ns'] = round(smoothed_prediction['main']['ns'], 1)
        smoothed_prediction['main']['eo'] = round(smoothed_prediction['main']['eo'], 1)
        smoothed_prediction['turn']['ns'] = round(smoothed_prediction['turn']['ns'], 1)
        smoothed_prediction['turn']['eo'] = round(smoothed_prediction['turn']['eo'], 1)
        
        return smoothed_prediction
    
    def should_update_prediction(self, vehicle_counts):
        """
        Determina si debemos actualizar la predicción basado en cambios en los datos
        
        Args:
            vehicle_counts: Nuevos conteos de vehículos
            
        Returns:
            bool: True si debemos actualizar la predicción
        """
        # Siempre actualizamos la primera vez
        if self.last_prediction is None:
            return True
        
        # Si han pasado más de 30 segundos desde la última actualización, actualizamos
        current_time = time.time()
        if current_time - self.last_update_time > 30:
            return True
            
        # Verificar si ha habido cambios significativos en los conteos
        if hasattr(self, 'last_vehicle_counts'):
            significant_change = False
            
            # Verificar cada dirección
            for direction in ['Norte', 'Sur', 'Este', 'Oeste']:
                current = vehicle_counts.get(direction, 0)
                previous = self.last_vehicle_counts.get(direction, 0)
                
                # Si hay un cambio de más de 2 vehículos, consideramos significativo
                if abs(current - previous) > 2:
                    significant_change = True
                    break
                    
            return significant_change
        
        # Primera vez, no hay conteos previos
        return True
    
    def predict_green_times(self, vehicle_counts):
        """
        Predice los tiempos óptimos de semáforo en verde basados en conteo vehicular
        
        Args:
            vehicle_counts: Diccionario con conteo por dirección {"Norte": n, "Sur": s, "Este": e, "Oeste": w}
        
        Returns:
            tuple: (predictions, cycle_sequence)
                - predictions: Dict con tiempos para flujos principales y giros
                - cycle_sequence: Lista de fases con sus tiempos y estados
        """
        # Verificar si debemos actualizar la predicción
        if not self.should_update_prediction(vehicle_counts):
            return self.last_prediction
            
        # Evitar procesamiento concurrente
        if self.is_processing:
            return self.last_prediction if self.last_prediction else self._fallback_prediction(vehicle_counts)
            
        self.is_processing = True
        self.last_vehicle_counts = vehicle_counts.copy()
        self.last_update_time = time.time()
            
        try:
            # Equalizar datos para que coincidan con el formato del dataset de entrenamiento
            equalized_data = self.equalize_dashboard_data(vehicle_counts)
            
            # Crear secuencia para LSTM
            input_sequence = self.create_sequence(equalized_data)
            
            # Si tenemos modelo entrenado, usarlo para predecir
            if self.model is not None:
                try:
                    # Hacer predicción con el modelo LSTM
                    prediction_scaled = self.model.predict(input_sequence, verbose=0)
                    
                    # Invertir el escalado para obtener tiempos reales
                    if hasattr(self, 'scaler_y') and self.scaler_y is not None:
                        try:
                            prediction = self.scaler_y.inverse_transform(prediction_scaled)
                            
                            # Extraer predicciones (asumiendo 4 valores de salida en este orden)
                            ns_straight_time = prediction[0][0]
                            ns_left_time = prediction[0][1]
                            ew_straight_time = prediction[0][2]
                            ew_left_time = prediction[0][3]
                            
                            # Limitar a valores razonables
                            ns_straight_time = max(10.0, min(60.0, ns_straight_time))
                            ns_left_time = max(5.0, min(30.0, ns_left_time))
                            ew_straight_time = max(10.0, min(60.0, ew_straight_time))
                            ew_left_time = max(5.0, min(30.0, ew_left_time))
                            
                            # Resultados como diccionario
                            raw_predictions = {
                                'main': {
                                    'ns': float(round(ns_straight_time, 1)),
                                    'eo': float(round(ew_straight_time, 1))
                                },
                                'turn': {
                                    'ns': float(round(ns_left_time, 1)),
                                    'eo': float(round(ew_left_time, 1))
                                }
                            }
                            
                            # Aplicar suavizado para evitar cambios bruscos
                            predictions = self.smooth_predictions(raw_predictions)
                            
                            # Calcular la secuencia del ciclo
                            cycle_sequence = self._calculate_cycle_sequence(predictions)
                            
                            self.last_prediction = (predictions, cycle_sequence)
                            self.is_processing = False
                            return self.last_prediction
                            
                        except Exception as e:
                            print(f"Error al invertir escalado: {e}")
                            # Usar método de fallback
                except Exception as e:
                    print(f"Error en predicción LSTM: {e}")
                    # Usar método de fallback
            
            # Método de fallback si no hay modelo o hay error
            result = self._fallback_prediction(vehicle_counts)
            self.is_processing = False
            return result
            
        except Exception as e:
            print(f"Error general en predicción: {e}")
            self.is_processing = False
            return self._fallback_prediction(vehicle_counts)
    
    def _fallback_prediction(self, vehicle_counts):
        """
        Método de fallback que simula predicciones basadas en reglas simples
        
        Args:
            vehicle_counts: Diccionario con conteo por dirección
            
        Returns:
            tuple: (predictions, cycle_sequence)
        """
        # Extraer conteos
        north = vehicle_counts.get("Norte", 0)
        south = vehicle_counts.get("Sur", 0)
        east = vehicle_counts.get("Este", 0)
        west = vehicle_counts.get("Oeste", 0)
        
        # Flujo Norte-Sur (suma de vehículos)
        ns_flow = north + south
        # Flujo Este-Oeste (suma de vehículos)
        ew_flow = east + west
        
        # Calculamos tiempo base + factor por vehículo
        base_time = 10.0  # segundos mínimos en verde
        factor = 1.2      # segundos adicionales por vehículo
        
        # Tiempo verde para movimiento recto
        ns_straight_time = max(base_time, base_time + factor * ns_flow)
        ew_straight_time = max(base_time, base_time + factor * ew_flow)
        
        # Tiempo para giros a la izquierda (30% del tiempo recto)
        ns_left_time = max(base_time, ns_straight_time * 0.3)
        ew_left_time = max(base_time, ew_straight_time * 0.3)
        
        # Tiempo mínimo de seguridad
        min_time = 5.0
        
        # Ajustar tiempo mínimo
        ns_straight_time = max(min_time, ns_straight_time)
        ew_straight_time = max(min_time, ew_straight_time)
        ns_left_time = max(min_time, ns_left_time)
        ew_left_time = max(min_time, ew_left_time)
        
        # Resultados como diccionario
        predictions = {
            'main': {
                'ns': round(ns_straight_time, 1),
                'eo': round(ew_straight_time, 1)
            },
            'turn': {
                'ns': round(ns_left_time, 1),
                'eo': round(ew_left_time, 1)
            }
        }
        
        # Suavizado si hay predicciones anteriores
        predictions = self.smooth_predictions(predictions)
        
        # Calculamos la secuencia de fases del ciclo
        cycle_sequence = self._calculate_cycle_sequence(predictions)
        
        # Guardar esta predicción para futuras comparaciones
        self.last_prediction = (predictions, cycle_sequence)
        return self.last_prediction
    
    def _calculate_cycle_sequence(self, predictions):
        """
        Calcula la secuencia de fases del ciclo completo
        
        Args:
            predictions: Diccionario con los tiempos de verde predichos
            
        Returns:
            list: Lista de fases con sus tiempos y estados
        """
        # Tiempos de transición/amarillo
        yellow_time = 3.0
        
        # Secuencia de fases (satisfaciendo la restricción de flujos perpendiculares)
        cycle = []
        
        # FASE 1: NS Recto + EO Giro izquierda
        cycle.append({
            'name': 'NS Recto + EO Giro',
            'duration': predictions['main']['ns'],
            'states': {
                'ns_straight': 'VERDE',
                'ns_left': 'ROJO',
                'eo_straight': 'ROJO',
                'eo_left': 'VERDE'
            }
        })
        
        # Transición amarilla
        cycle.append({
            'name': 'Transición 1',
            'duration': yellow_time,
            'states': {
                'ns_straight': 'AMARILLO',
                'ns_left': 'ROJO',
                'eo_straight': 'ROJO',
                'eo_left': 'AMARILLO'
            }
        })
        
        # FASE 2: NS Giro izquierda + EO Recto
        cycle.append({
            'name': 'NS Giro + EO Recto',
            'duration': predictions['main']['eo'],
            'states': {
                'ns_straight': 'ROJO',
                'ns_left': 'VERDE',
                'eo_straight': 'VERDE',
                'eo_left': 'ROJO'
            }
        })
        
        # Transición amarilla
        cycle.append({
            'name': 'Transición 2',
            'duration': yellow_time,
            'states': {
                'ns_straight': 'ROJO',
                'ns_left': 'AMARILLO',
                'eo_straight': 'AMARILLO',
                'eo_left': 'ROJO'
            }
        })
        
        return cycle

class TrafficPredictor:
    """
    Clase compatibilidad para uso actual en dashboard
    """
    def __init__(self):
        self.lstm_predictor = LSTMTrafficPredictor()
        
    def predict_green_times(self, vehicle_counts):
        return self.lstm_predictor.predict_green_times(vehicle_counts)