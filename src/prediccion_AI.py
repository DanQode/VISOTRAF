import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from collections import deque
from datetime import datetime
import os
from src.prediccion_AI import TrafficPredictor

class TrafficPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.last_timestamp = None
        self.historical_counts = {
            'Norte': deque(maxlen=6),
            'Sur': deque(maxlen=6),
            'Este': deque(maxlen=6),
            'Oeste': deque(maxlen=6)
        }
        # Definir tiempos límite
        self.MIN_GREEN = 20    
        self.MAX_GREEN = 90    
        self.MIN_TURN = 15     
        self.MAX_TURN = 30     

    def update_counts(self, counts):
        """
        Actualiza los conteos históricos con nuevos datos
        
        Parameters:
        -----------
        counts : dict
            Diccionario con conteos actuales {'Norte': n, 'Sur': n, 'Este': n, 'Oeste': n}
        """
        timestamp = datetime.now()
        if self.last_timestamp is None or (timestamp - self.last_timestamp).seconds >= 10:
            self.last_timestamp = timestamp
            for direction, count in counts.items():
                self.historical_counts[direction].append(count)
            return True
        return False

    def predict_green_times(self, current_counts):
        """
        Predice tiempos óptimos usando los conteos actuales
        """
        # Actualizar histórico
        self.update_counts(current_counts)
        
        # Calcular flujos
        flow_rates = self.calculate_flow_rates()
        
        # Flujos agrupados N-S y E-O
        ns_flow = flow_rates['Norte'] + flow_rates['Sur']
        eo_flow = flow_rates['Este'] + flow_rates['Oeste']
        
        # Calcular tiempos
        total_flow = ns_flow + eo_flow
        if total_flow > 0:
            ns_ratio = ns_flow / total_flow
            eo_ratio = eo_flow / total_flow
            
            base_ns = np.clip(self.MIN_GREEN + (self.MAX_GREEN - self.MIN_GREEN) * ns_ratio, 
                            self.MIN_GREEN, self.MAX_GREEN)
            base_eo = np.clip(self.MIN_GREEN + (self.MAX_GREEN - self.MIN_GREEN) * eo_ratio,
                            self.MIN_GREEN, self.MAX_GREEN)
            
            predictions = {
                'main': {
                    'ns': base_ns,
                    'eo': base_eo
                },
                'turn': {
                    'ns': np.clip(self.MIN_TURN + (self.MAX_TURN - self.MIN_TURN) * 
                                (current_counts['Norte'] + current_counts['Sur']) / 30,
                                self.MIN_TURN, self.MAX_TURN),
                    'eo': np.clip(self.MIN_TURN + (self.MAX_TURN - self.MIN_TURN) * 
                                (current_counts['Este'] + current_counts['Oeste']) / 30,
                                self.MIN_TURN, self.MAX_TURN)
                }
            }
        else:
            predictions = {
                'main': {'ns': self.MIN_GREEN, 'eo': self.MIN_GREEN},
                'turn': {'ns': self.MIN_TURN, 'eo': self.MIN_TURN}
            }
        
        return predictions, self.calculate_cycle_sequence(predictions)

    def calculate_flow_rates(self):
        """Calcula tasas de flujo basadas en los últimos 6 conteos"""
        flow_rates = {}
        for direction, counts in self.historical_counts.items():
            if counts:
                avg_count = sum(counts) / len(counts)
                flow_rates[direction] = int(avg_count * 360)
            else:
                flow_rates[direction] = 0
        return flow_rates

    def calculate_cycle_sequence(self, predictions):
        """Calcula la secuencia completa del ciclo"""
        YELLOW_TIME = 3
        ALL_RED = 2
        
        cycle_sequence = [
            {
                'phase': 'NS_MAIN_AND_TURN',
                'duration': predictions['main']['ns'],
                'description': 'Verde principal y giro N-S',
                'states': {
                    'ns_main': 'GREEN',
                    'ns_turn': 'GREEN',
                    'eo_main': 'RED',
                    'eo_turn': 'RED'
                }
            },
            {
                'phase': 'NS_YELLOW',
                'duration': YELLOW_TIME,
                'description': 'Ámbar N-S',
                'states': {
                    'ns_main': 'YELLOW',
                    'ns_turn': 'YELLOW',
                    'eo_main': 'RED',
                    'eo_turn': 'RED'
                }
            },
            {
                'phase': 'ALL_RED_1',
                'duration': ALL_RED,
                'description': 'Todo Rojo',
                'states': {
                    'ns_main': 'RED',
                    'ns_turn': 'RED',
                    'eo_main': 'RED',
                    'eo_turn': 'RED'
                }
            },
            {
                'phase': 'EO_MAIN_AND_TURN',
                'duration': predictions['main']['eo'],
                'description': 'Verde principal y giro E-O',
                'states': {
                    'ns_main': 'RED',
                    'ns_turn': 'RED',
                    'eo_main': 'GREEN',
                    'eo_turn': 'GREEN'
                }
            },
            {
                'phase': 'EO_YELLOW',
                'duration': YELLOW_TIME,
                'description': 'Ámbar E-O',
                'states': {
                    'ns_main': 'RED',
                    'ns_turn': 'RED',
                    'eo_main': 'YELLOW',
                    'eo_turn': 'YELLOW'
                }
            },
            {
                'phase': 'ALL_RED_2',
                'duration': ALL_RED,
                'description': 'Todo Rojo',
                'states': {
                    'ns_main': 'RED',
                    'ns_turn': 'RED',
                    'eo_main': 'RED',
                    'eo_turn': 'RED'
                }
            }
        ]
        return cycle_sequence