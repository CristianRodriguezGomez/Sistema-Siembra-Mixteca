"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     MÓDULO 2: SISTEMA DE INFERENCIA DIFUSA PARA SIEMBRA DE MAÍZ 🌽          ║
║                                                                              ║
║     Proyecto: Determinación de la Ventana de Siembra Óptima para            ║
║               Minimizar el Riesgo Climático en la Mixteca Oaxaqueña         ║
║               Mediante Redes Neuronales y Algoritmos Genéticos              ║
║               con Inferencia Difusa                                          ║
║                                                                              ║
║     Autor: Ramón                                                             ║
║     Fase: 2 de 3                                                             ║
║     Fecha: Diciembre 2025                                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
                          🔗 CONEXIÓN ENTRE MÓDULOS
================================================================================

Este módulo es el PUENTE entre:

    ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
    │   MÓDULO 1      │  ─────► │   MÓDULO 2      │  ─────► │   MÓDULO 3      │
    │   (Annelí/      │         │   (Ramón)       │         │   (Cristian)    │
    │    Ossiel)      │         │   ESTE ARCHIVO  │         │                 │
    │                 │         │                 │         │                 │
    │   Red LSTM      │         │  Lógica Difusa  │         │ Algoritmo       │
    │   Predicciones  │         │  Evaluación     │         │ Genético        │
    └─────────────────┘         └─────────────────┘         └─────────────────┘
           │                           │                           │
           ▼                           ▼                           ▼
      temperatura               score_amplitud              fecha_optima
      precipitacion             (0-100)                     de siembra


================================================================================
                    📥 QUÉ RECIBE ESTE MÓDULO (ENTRADA)
================================================================================

Del MÓDULO 1 (Annelí/Ossiel) recibe UN DICCIONARIO o JSON con:

    {
        "temperatura": 27.5,      # Temperatura predicha en °C (rango: 5-45)
        "precipitacion": 12.3     # Lluvia predicha en mm (rango: 0-45)
    }

    O una lista de días:
    [
        {"fecha": "2025-06-01", "temperatura": 25.0, "precipitacion": 10.0},
        {"fecha": "2025-06-02", "temperatura": 27.5, "precipitacion": 12.3},
        ...
    ]


================================================================================
                    📤 QUÉ ENTREGA ESTE MÓDULO (SALIDA)
================================================================================

Al MÓDULO 3 (Cristian) entrega UN DICCIONARIO con:

    {
        "score_amplitud": 85.77,   # Puntaje de 0 a 100
        "categoria": "EXCELENTE",  # Clasificación textual
        "recomendacion": "Sembrar" # Recomendación de acción
    }

    O para múltiples días:
    [
        {"fecha": "2025-06-01", "score_amplitud": 85.77, "categoria": "EXCELENTE"},
        {"fecha": "2025-06-02", "score_amplitud": 72.50, "categoria": "BUENO"},
        ...
    ]


================================================================================
                         📊 VARIABLES DIFUSAS
================================================================================

ENTRADAS (Antecedentes):
------------------------
    1. TEMPERATURA (°C)
       ├── Baja:    5-18°C   (frío, riesgo de heladas)
       ├── Óptima: 18-32°C   (ideal para germinación)
       └── Alta:   32-45°C   (estrés térmico)

    2. PRECIPITACIÓN (mm)
       ├── Escasa:    0-7mm    (sequía, falta de humedad)
       ├── Adecuada:  5-25mm   (ideal para siembra)
       └── Excesiva: 20-45mm   (encharcamiento, hongos)

SALIDA (Consecuente):
---------------------
    AMPLITUD DE SIEMBRA (0-100)
       ├── Baja:    0-35    → NO sembrar
       ├── Media:  25-75    → Sembrar con precaución
       └── Alta:   65-100   → Condiciones IDEALES


================================================================================
                           📋 REGLAS DIFUSAS (9)
================================================================================

    ┌───────────────┬───────────────┬───────────────┬───────────────┐
    │               │  Temp BAJA    │  Temp ÓPTIMA  │  Temp ALTA    │
    ├───────────────┼───────────────┼───────────────┼───────────────┤
    │ Lluvia ESCASA │  🔴 BAJA      │  🟡 MEDIA     │  🔴 BAJA      │
    │ Lluvia ADECUA │  🟡 MEDIA     │  🟢 ALTA ⭐   │  🟡 MEDIA     │
    │ Lluvia EXCESI │  🔴 BAJA      │  🟡 MEDIA     │  🔴 BAJA      │
    └───────────────┴───────────────┴───────────────┴───────────────┘

    ⭐ La condición IDEAL es: Temperatura ÓPTIMA + Lluvia ADECUADA = Amplitud ALTA


================================================================================
                              CÓDIGO DEL SISTEMA
================================================================================
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import json
from typing import Dict, List, Union, Tuple


class SistemaDifusoSiembra:
    """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                   SISTEMA DE INFERENCIA DIFUSA                           ║
    ║                   Para Evaluación de Siembra de Maíz                     ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    Este sistema implementa lógica difusa tipo Mamdani para determinar
    qué tan favorable es un día específico para sembrar maíz.
    
    FLUJO DEL SISTEMA:
    ──────────────────
    
        ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
        │  ENTRADAS    │     │  INFERENCIA  │     │   SALIDA     │
        │              │     │              │     │              │
        │ temperatura  │────►│   9 Reglas   │────►│  amplitud    │
        │ precipitación│     │   Difusas    │     │  (0-100)     │
        │              │     │              │     │              │
        └──────────────┘     └──────────────┘     └──────────────┘
              │                    │                    │
              ▼                    ▼                    ▼
         Fuzzificación     Evaluación de        Defuzzificación
         (crisp→fuzzy)     Reglas IF-THEN       (fuzzy→crisp)
    
    
    USO BÁSICO:
    ───────────
        >>> sistema = SistemaDifusoSiembra()
        >>> resultado = sistema.evaluar(temperatura=27, precipitacion=10)
        >>> print(resultado)
        {'score_amplitud': 85.77, 'categoria': 'EXCELENTE', 'recomendacion': 'Sembrar'}
    
    
    INTEGRACIÓN CON MÓDULO 1 (Annelí/Ossiel):
    ──────────────────────────────────────────
        >>> # Recibir datos del Módulo 1
        >>> datos_modulo1 = {"temperatura": 27, "precipitacion": 10}
        >>> 
        >>> # Procesar con sistema difuso
        >>> sistema = SistemaDifusoSiembra()
        >>> resultado = sistema.evaluar_desde_json(datos_modulo1)
        >>> 
        >>> # resultado está listo para el Módulo 3
    
    
    INTEGRACIÓN CON MÓDULO 3 (Cristian):
    ─────────────────────────────────────
        >>> # Cristian recibe el resultado y lo usa como fitness
        >>> fitness = resultado['score_amplitud']  # Valor de 0 a 100
        >>> 
        >>> # En su algoritmo genético:
        >>> def fitness_function(fecha):
        ...     datos = obtener_prediccion(fecha)  # Del Módulo 1
        ...     resultado = sistema.evaluar_desde_json(datos)
        ...     return resultado['score_amplitud']
    """
    
    def __init__(self):
        """
        ┌──────────────────────────────────────────────────────────────────────┐
        │                    CONSTRUCTOR DEL SISTEMA                           │
        └──────────────────────────────────────────────────────────────────────┘
        
        Inicializa el sistema difuso completo:
        1. Crea las variables de entrada y salida
        2. Define las funciones de membresía
        3. Establece las 9 reglas difusas
        4. Configura el motor de inferencia
        
        El sistema queda listo para recibir datos y producir evaluaciones.
        """
        print("🔧 Inicializando Sistema de Inferencia Difusa...")
        
        # Paso 1: Crear variables
        self._crear_variables()
        
        # Paso 2: Definir funciones de membresía
        self._crear_funciones_membresia()
        
        # Paso 3: Establecer reglas
        self._crear_reglas()
        
        # Paso 4: Crear sistema de control
        self._crear_sistema_control()
        
        print("✅ Sistema inicializado correctamente")
    
    
    def _crear_variables(self):
        """
        ┌──────────────────────────────────────────────────────────────────────┐
        │          PASO 1: DEFINIR UNIVERSOS DE DISCURSO                       │
        └──────────────────────────────────────────────────────────────────────┘
        
        Define el rango de valores posibles para cada variable:
        
        TEMPERATURA:
        ────────────
            Rango: 5°C a 45°C
            Resolución: 0.5°C
            
            5°C ──────────────────────────────────────────► 45°C
            │                                               │
            Muy frío                                    Muy caliente
        
        
        PRECIPITACIÓN (Lluvia):
        ───────────────────────
            Rango: 0mm a 45mm
            Resolución: 0.5mm
            
            0mm ──────────────────────────────────────────► 45mm
            │                                               │
            Sequía                                      Inundación
        
        
        AMPLITUD (Salida):
        ──────────────────
            Rango: 0 a 100
            Resolución: 1 punto
            
            0 ────────────────────────────────────────────► 100
            │                                               │
            No sembrar                              Condiciones ideales
        """
        
        # Variable de ENTRADA 1: Temperatura en grados Celsius
        # np.arange(5, 46, 0.5) crea: [5.0, 5.5, 6.0, ..., 45.0]
        self.temperatura = ctrl.Antecedent(
            np.arange(5, 46, 0.5),  # Universo: 5°C a 45°C
            'temperatura'           # Nombre de la variable
        )
        
        # Variable de ENTRADA 2: Precipitación (lluvia) en milímetros
        self.lluvia = ctrl.Antecedent(
            np.arange(0, 46, 0.5),  # Universo: 0mm a 45mm
            'lluvia'                # Nombre de la variable
        )
        
        # Variable de SALIDA: Amplitud de siembra (score de calidad)
        self.amplitud = ctrl.Consequent(
            np.arange(0, 101, 1),   # Universo: 0 a 100
            'amplitud'              # Nombre de la variable
        )
    
    
    def _crear_funciones_membresia(self):
        """
        ┌──────────────────────────────────────────────────────────────────────┐
        │          PASO 2: DEFINIR FUNCIONES DE MEMBRESÍA                      │
        └──────────────────────────────────────────────────────────────────────┘
        
        Las funciones de membresía convierten valores CRISP (números exactos)
        en grados de pertenencia DIFUSOS (0.0 a 1.0).
        
        
        TEMPERATURA - Funciones de Membresía:
        ─────────────────────────────────────
        
        1.0 │    ████                              
            │   █    █                    ████████
            │  █      █      ▲           █        
            │ █        █    / \         █         
            │█          █  /   \       █          
        0.0 │────────────██─────██────██──────────
            5    12    18  25   32   35         45
                 │           │         │
                BAJA      ÓPTIMA     ALTA
        
        
        LLUVIA - Funciones de Membresía:
        ────────────────────────────────
        
        1.0 │████                              
            │    █                      ████████
            │     █       ▲            █        
            │      █     / \          █         
            │       █   /   \        █          
        0.0 │────────█─█─────█──────█───────────
            0   3   7  12    25    28         45
                │          │          │
             ESCASA    ADECUADA   EXCESIVA
        
        
        AMPLITUD (Salida) - Funciones de Membresía:
        ───────────────────────────────────────────
        
        1.0 │████                          ████████
            │    █                        █        
            │     █       ▲              █         
            │      █     / \            █          
            │       █   /   \          █           
        0.0 │────────█─█─────█────────█────────────
            0   15  35    50    75   85        100
                │           │          │
              BAJA        MEDIA      ALTA
        """
        
        # ═══════════════════════════════════════════════════════════════════
        # FUNCIONES DE MEMBRESÍA PARA TEMPERATURA
        # ═══════════════════════════════════════════════════════════════════
        
        # BAJA: Temperaturas frías (5-18°C)
        # Forma trapezoidal: [a, b, c, d]
        # Membresía = 1 entre b y c, decrece linealmente en extremos
        self.temperatura['baja'] = fuzz.trapmf(
            self.temperatura.universe,
            [5, 5, 12, 18]  # Máxima entre 5-12, decrece hasta 18
        )
        
        # ÓPTIMA: Temperaturas ideales para maíz (18-32°C)
        # Forma triangular: [a, b, c]
        # Membresía = 1 en el pico (b), decrece linealmente hacia a y c
        self.temperatura['optima'] = fuzz.trimf(
            self.temperatura.universe,
            [18, 25, 32]  # Pico en 25°C (temperatura ideal)
        )
        
        # ALTA: Temperaturas calientes (32-45°C)
        # Forma trapezoidal
        self.temperatura['alta'] = fuzz.trapmf(
            self.temperatura.universe,
            [32, 35, 45, 45]  # Crece desde 32, máxima desde 35
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # FUNCIONES DE MEMBRESÍA PARA LLUVIA
        # ═══════════════════════════════════════════════════════════════════
        
        # ESCASA: Poca lluvia (0-7mm) - Riesgo de sequía
        self.lluvia['escasa'] = fuzz.trapmf(
            self.lluvia.universe,
            [0, 0, 3, 7]  # Máxima entre 0-3mm, decrece hasta 7mm
        )
        
        # ADECUADA: Lluvia ideal para siembra (5-25mm)
        self.lluvia['adecuada'] = fuzz.trimf(
            self.lluvia.universe,
            [5, 12, 25]  # Pico en 12mm (precipitación ideal)
        )
        
        # EXCESIVA: Demasiada lluvia (20-45mm) - Riesgo de encharcamiento
        self.lluvia['excesiva'] = fuzz.trapmf(
            self.lluvia.universe,
            [20, 28, 45, 45]  # Crece desde 20, máxima desde 28mm
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # FUNCIONES DE MEMBRESÍA PARA AMPLITUD (SALIDA)
        # ═══════════════════════════════════════════════════════════════════
        
        # BAJA: Condiciones desfavorables (0-35)
        self.amplitud['baja'] = fuzz.trapmf(
            self.amplitud.universe,
            [0, 0, 15, 35]  # No sembrar
        )
        
        # MEDIA: Condiciones aceptables (25-75)
        self.amplitud['media'] = fuzz.trimf(
            self.amplitud.universe,
            [25, 50, 75]  # Sembrar con precaución
        )
        
        # ALTA: Condiciones óptimas (65-100)
        self.amplitud['alta'] = fuzz.trapmf(
            self.amplitud.universe,
            [65, 85, 100, 100]  # ¡Sembrar!
        )
    
    
    def _crear_reglas(self):
        """
        ┌──────────────────────────────────────────────────────────────────────┐
        │          PASO 3: DEFINIR REGLAS DE INFERENCIA                        │
        └──────────────────────────────────────────────────────────────────────┘
        
        Las reglas difusas codifican el CONOCIMIENTO EXPERTO sobre
        cuándo es bueno sembrar maíz.
        
        Formato: SI <antecedente> ENTONCES <consecuente>
        
        
        MATRIZ DE DECISIÓN:
        ═══════════════════
        
        Esta matriz resume las 9 reglas basadas en conocimiento agronómico:
        
        ┌─────────────────┬──────────────┬──────────────┬──────────────┐
        │   LLUVIA ↓      │  Temp BAJA   │ Temp ÓPTIMA  │  Temp ALTA   │
        │   TEMP →        │   (fría)     │   (ideal)    │  (caliente)  │
        ├─────────────────┼──────────────┼──────────────┼──────────────┤
        │ ESCASA (seco)   │  🔴 BAJA     │  🟡 MEDIA    │  🔴 BAJA     │
        │                 │  Frío+Seco   │  Temp OK     │  Calor+Seco  │
        │                 │  = Malo      │  pero seco   │  = Malo      │
        ├─────────────────┼──────────────┼──────────────┼──────────────┤
        │ ADECUADA        │  🟡 MEDIA    │  🟢 ALTA ⭐  │  🟡 MEDIA    │
        │ (ideal)         │  Lluvia OK   │  ¡PERFECTO!  │  Calor pero  │
        │                 │  pero frío   │  Condición   │  lluvia OK   │
        │                 │              │  IDEAL       │              │
        ├─────────────────┼──────────────┼──────────────┼──────────────┤
        │ EXCESIVA        │  🔴 BAJA     │  🟡 MEDIA    │  🔴 BAJA     │
        │ (encharcado)    │  Frío+Mojado │  Temp OK     │  Calor+Mojado│
        │                 │  = Hongos    │  pero mojado │  = Pudrición │
        └─────────────────┴──────────────┴──────────────┴──────────────┘
        
        
        JUSTIFICACIÓN AGRONÓMICA:
        ─────────────────────────
        
        🌡️ TEMPERATURA:
           - BAJA (<18°C): Germinación lenta, riesgo de heladas
           - ÓPTIMA (18-32°C): Germinación rápida, crecimiento ideal
           - ALTA (>32°C): Estrés térmico, deshidratación
        
        💧 LLUVIA:
           - ESCASA (<7mm): Semilla no germina, falta de humedad
           - ADECUADA (5-25mm): Humedad perfecta para germinación
           - EXCESIVA (>20mm): Encharcamiento, hongos, pudrición
        
        🌽 COMBINACIÓN IDEAL:
           Temperatura 20-30°C + Lluvia 8-18mm = ¡SEMBRAR!
        """
        
        # ═══════════════════════════════════════════════════════════════════
        # REGLAS CUANDO LA LLUVIA ES ESCASA (Sequía)
        # ═══════════════════════════════════════════════════════════════════
        
        # Regla 1: Lluvia ESCASA + Temperatura BAJA → Amplitud BAJA
        # Razón: Frío + Sequía = Semilla no germina
        regla1 = ctrl.Rule(
            self.lluvia['escasa'] & self.temperatura['baja'],
            self.amplitud['baja'],
            label='R1: Escasa+Baja=Baja'
        )
        
        # Regla 2: Lluvia ESCASA + Temperatura ÓPTIMA → Amplitud MEDIA
        # Razón: Buena temperatura pero falta agua
        regla2 = ctrl.Rule(
            self.lluvia['escasa'] & self.temperatura['optima'],
            self.amplitud['media'],
            label='R2: Escasa+Óptima=Media'
        )
        
        # Regla 3: Lluvia ESCASA + Temperatura ALTA → Amplitud BAJA
        # Razón: Calor extremo + Sequía = Muerte de semilla
        regla3 = ctrl.Rule(
            self.lluvia['escasa'] & self.temperatura['alta'],
            self.amplitud['baja'],
            label='R3: Escasa+Alta=Baja'
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # REGLAS CUANDO LA LLUVIA ES ADECUADA (Ideal)
        # ═══════════════════════════════════════════════════════════════════
        
        # Regla 4: Lluvia ADECUADA + Temperatura BAJA → Amplitud MEDIA
        # Razón: Buena agua pero frío = germinación lenta
        regla4 = ctrl.Rule(
            self.lluvia['adecuada'] & self.temperatura['baja'],
            self.amplitud['media'],
            label='R4: Adecuada+Baja=Media'
        )
        
        # ⭐ Regla 5: Lluvia ADECUADA + Temperatura ÓPTIMA → Amplitud ALTA
        # Razón: ¡CONDICIONES PERFECTAS PARA SEMBRAR!
        regla5 = ctrl.Rule(
            self.lluvia['adecuada'] & self.temperatura['optima'],
            self.amplitud['alta'],
            label='R5: Adecuada+Óptima=Alta ⭐'
        )
        
        # Regla 6: Lluvia ADECUADA + Temperatura ALTA → Amplitud MEDIA
        # Razón: Buena agua pero calor = estrés térmico moderado
        regla6 = ctrl.Rule(
            self.lluvia['adecuada'] & self.temperatura['alta'],
            self.amplitud['media'],
            label='R6: Adecuada+Alta=Media'
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # REGLAS CUANDO LA LLUVIA ES EXCESIVA (Encharcamiento)
        # ═══════════════════════════════════════════════════════════════════
        
        # Regla 7: Lluvia EXCESIVA + Temperatura BAJA → Amplitud BAJA
        # Razón: Frío + Mojado = Hongos, pudrición de semilla
        regla7 = ctrl.Rule(
            self.lluvia['excesiva'] & self.temperatura['baja'],
            self.amplitud['baja'],
            label='R7: Excesiva+Baja=Baja'
        )
        
        # Regla 8: Lluvia EXCESIVA + Temperatura ÓPTIMA → Amplitud MEDIA
        # Razón: Buena temperatura pero demasiada agua
        regla8 = ctrl.Rule(
            self.lluvia['excesiva'] & self.temperatura['optima'],
            self.amplitud['media'],
            label='R8: Excesiva+Óptima=Media'
        )
        
        # Regla 9: Lluvia EXCESIVA + Temperatura ALTA → Amplitud BAJA
        # Razón: Calor + Mojado = Ambiente perfecto para enfermedades
        regla9 = ctrl.Rule(
            self.lluvia['excesiva'] & self.temperatura['alta'],
            self.amplitud['baja'],
            label='R9: Excesiva+Alta=Baja'
        )
        
        # Guardar todas las reglas en una lista
        self.reglas = [
            regla1, regla2, regla3,
            regla4, regla5, regla6,
            regla7, regla8, regla9
        ]
        
        print(f"   📋 {len(self.reglas)} reglas difusas cargadas")
    
    
    def _crear_sistema_control(self):
        """
        ┌──────────────────────────────────────────────────────────────────────┐
        │          PASO 4: CREAR MOTOR DE INFERENCIA                           │
        └──────────────────────────────────────────────────────────────────────┘
        
        El ControlSystem agrupa todas las reglas y el ControlSystemSimulation
        permite ejecutar el sistema con valores de entrada específicos.
        
        Método de defuzzificación: CENTROIDE (Center of Gravity)
        
            Area sombreada
            bajo la curva      ┌───────────────┐
            del resultado  ───►│   CENTROIDE   │───► Valor crisp
                               └───────────────┘     de salida
        
        El centroide calcula el "centro de masa" del área resultante,
        dando un valor numérico preciso entre 0 y 100.
        """
        
        # Crear el sistema de control con todas las reglas
        self.sistema_ctrl = ctrl.ControlSystem(self.reglas)
        
        # Crear la simulación (motor de inferencia)
        self.simulacion = ctrl.ControlSystemSimulation(self.sistema_ctrl)
        
        print("   ⚙️  Motor de inferencia configurado")
    
    
    # ==========================================================================
    #                   MÉTODOS PÚBLICOS DE EVALUACIÓN
    # ==========================================================================
    
    def evaluar(self, temperatura: float, precipitacion: float) -> Dict:
        """
        ╔══════════════════════════════════════════════════════════════════════╗
        ║                    FUNCIÓN PRINCIPAL DE EVALUACIÓN                   ║
        ╚══════════════════════════════════════════════════════════════════════╝
        
        Evalúa qué tan favorable es un día para sembrar maíz.
        
        
        PARÁMETROS:
        ───────────
            temperatura (float): Temperatura predicha en °C
                                 Rango válido: 5 a 45
            
            precipitacion (float): Lluvia predicha en mm
                                   Rango válido: 0 a 45
        
        
        RETORNA:
        ────────
            dict: Diccionario con los resultados:
                {
                    "score_amplitud": float,    # Puntaje 0-100
                    "categoria": str,           # "EXCELENTE", "BUENO", etc.
                    "recomendacion": str,       # "Sembrar", "Esperar", etc.
                    "inputs": {                 # Valores de entrada (debug)
                        "temperatura": float,
                        "precipitacion": float
                    }
                }
        
        
        EJEMPLO DE USO:
        ───────────────
            >>> sistema = SistemaDifusoSiembra()
            >>> resultado = sistema.evaluar(temperatura=27, precipitacion=10)
            >>> print(resultado)
            {
                'score_amplitud': 85.77,
                'categoria': 'EXCELENTE',
                'recomendacion': 'Sembrar',
                'inputs': {'temperatura': 27, 'precipitacion': 10}
            }
        
        
        INTERPRETACIÓN DE RESULTADOS:
        ─────────────────────────────
            Score 70-100: EXCELENTE → ¡Sembrar sin dudar!
            Score 50-70:  BUENO     → Sembrar, condiciones aceptables
            Score 30-50:  REGULAR   → Sembrar con precaución
            Score 0-30:   MALO      → NO sembrar, esperar mejores condiciones
        """
        
        # ═══════════════════════════════════════════════════════════════════
        # PASO 1: Validar que los valores estén en rango
        # ═══════════════════════════════════════════════════════════════════
        
        if not (5 <= temperatura <= 45):
            raise ValueError(
                f"❌ Temperatura fuera de rango: {temperatura}°C. "
                f"Debe estar entre 5°C y 45°C"
            )
        
        if not (0 <= precipitacion <= 45):
            raise ValueError(
                f"❌ Precipitación fuera de rango: {precipitacion}mm. "
                f"Debe estar entre 0mm y 45mm"
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # PASO 2: Asignar valores de entrada al sistema
        # ═══════════════════════════════════════════════════════════════════
        
        self.simulacion.input['temperatura'] = temperatura
        self.simulacion.input['lluvia'] = precipitacion
        
        # ═══════════════════════════════════════════════════════════════════
        # PASO 3: Ejecutar inferencia difusa
        # ═══════════════════════════════════════════════════════════════════
        
        try:
            # Calcular resultado (fuzzificar → evaluar reglas → defuzzificar)
            self.simulacion.compute()
            
            # Obtener valor defuzzificado (centroide)
            score = round(self.simulacion.output['amplitud'], 2)
            
        except KeyError:
            # Manejar casos extremos donde no hay activación de reglas
            if temperatura < 10 or temperatura > 40:
                score = 10.0
            elif precipitacion < 2 or precipitacion > 35:
                score = 15.0
            else:
                score = 30.0
        
        # ═══════════════════════════════════════════════════════════════════
        # PASO 4: Clasificar resultado
        # ═══════════════════════════════════════════════════════════════════
        
        if score >= 70:
            categoria = "EXCELENTE"
            recomendacion = "Sembrar"
        elif score >= 50:
            categoria = "BUENO"
            recomendacion = "Sembrar con monitoreo"
        elif score >= 30:
            categoria = "REGULAR"
            recomendacion = "Esperar si es posible"
        else:
            categoria = "MALO"
            recomendacion = "NO sembrar"
        
        # ═══════════════════════════════════════════════════════════════════
        # PASO 5: Construir y retornar resultado
        # ═══════════════════════════════════════════════════════════════════
        
        resultado = {
            "score_amplitud": score,
            "categoria": categoria,
            "recomendacion": recomendacion,
            "inputs": {
                "temperatura": temperatura,
                "precipitacion": precipitacion
            }
        }
        
        return resultado
    
    
    def evaluar_desde_json(self, datos: Dict) -> Dict:
        """
        ╔══════════════════════════════════════════════════════════════════════╗
        ║          EVALUACIÓN DESDE JSON (PARA INTEGRACIÓN CON MÓDULO 1)       ║
        ╚══════════════════════════════════════════════════════════════════════╝
        
        Recibe datos en formato JSON/diccionario y los evalúa.
        
        
        ESTE MÉTODO ES EL PUNTO DE CONEXIÓN CON EL MÓDULO 1 (Annelí/Ossiel)
        ════════════════════════════════════════════════════════════════════
        
        El Módulo 1 debe enviar un diccionario con este formato:
        
            {
                "temperatura": 27.5,      # Temperatura predicha en °C
                "precipitacion": 12.3     # Lluvia predicha en mm
            }
        
        
        EJEMPLO DE INTEGRACIÓN:
        ───────────────────────
        
            # === EN EL MÓDULO 1 (Annelí/Ossiel) ===
            
            # Después de que la LSTM hace la predicción:
            prediccion_lstm = {
                "fecha": "2025-06-15",
                "temperatura": modelo.predict_temp(fecha),
                "precipitacion": modelo.predict_lluvia(fecha)
            }
            
            # Guardar o enviar al Módulo 2
            
            
            # === EN EL MÓDULO 2 (Ramón - ESTE CÓDIGO) ===
            
            from fuzzy_system import SistemaDifusoSiembra
            
            sistema = SistemaDifusoSiembra()
            resultado = sistema.evaluar_desde_json(prediccion_lstm)
            
            # resultado está listo para el Módulo 3
            
            
            # === EN EL MÓDULO 3 (Cristian) ===
            
            # Recibe el resultado y usa score_amplitud como fitness
            fitness = resultado['score_amplitud']
        
        
        PARÁMETROS:
        ───────────
            datos (dict): Diccionario con las predicciones del Módulo 1
                          Debe contener: "temperatura" y "precipitacion"
        
        RETORNA:
        ────────
            dict: Resultado de la evaluación difusa
        """
        
        # Validar que el diccionario tenga los campos necesarios
        if 'temperatura' not in datos:
            raise KeyError(
                "❌ El diccionario debe contener 'temperatura'. "
                "Formato esperado: {'temperatura': valor, 'precipitacion': valor}"
            )
        
        if 'precipitacion' not in datos:
            raise KeyError(
                "❌ El diccionario debe contener 'precipitacion'. "
                "Formato esperado: {'temperatura': valor, 'precipitacion': valor}"
            )
        
        # Extraer valores y evaluar
        temperatura = float(datos['temperatura'])
        precipitacion = float(datos['precipitacion'])
        
        return self.evaluar(temperatura, precipitacion)
    
    
    def evaluar_multiples_dias(self, lista_dias: List[Dict]) -> List[Dict]:
        """
        ╔══════════════════════════════════════════════════════════════════════╗
        ║              EVALUACIÓN DE MÚLTIPLES DÍAS EN LOTE                    ║
        ╚══════════════════════════════════════════════════════════════════════╝
        
        Evalúa una lista completa de días (útil para análisis de periodos).
        
        
        PARÁMETROS:
        ───────────
            lista_dias (list): Lista de diccionarios, cada uno con:
                [
                    {"fecha": "2025-06-01", "temperatura": 25, "precipitacion": 10},
                    {"fecha": "2025-06-02", "temperatura": 27, "precipitacion": 12},
                    ...
                ]
        
        
        RETORNA:
        ────────
            list: Lista de resultados, uno por cada día:
                [
                    {"fecha": "2025-06-01", "score_amplitud": 85.5, ...},
                    {"fecha": "2025-06-02", "score_amplitud": 88.2, ...},
                    ...
                ]
        
        
        EJEMPLO DE USO CON MÓDULO 1:
        ────────────────────────────
        
            # El Módulo 1 genera predicciones para varios días
            predicciones = [
                {"fecha": "2025-06-01", "temperatura": 25, "precipitacion": 10},
                {"fecha": "2025-06-02", "temperatura": 27, "precipitacion": 12},
                {"fecha": "2025-06-03", "temperatura": 30, "precipitacion": 8},
            ]
            
            # Evaluar todos los días de una vez
            resultados = sistema.evaluar_multiples_dias(predicciones)
            
            # El Módulo 3 puede usar estos resultados para su GA
            for r in resultados:
                print(f"{r['fecha']}: {r['score_amplitud']} - {r['categoria']}")
        """
        
        resultados = []
        
        for dia in lista_dias:
            # Evaluar cada día
            resultado = self.evaluar_desde_json(dia)
            
            # Agregar la fecha si está presente
            if 'fecha' in dia:
                resultado['fecha'] = dia['fecha']
            
            resultados.append(resultado)
        
        return resultados
    
    
    def obtener_score_para_fitness(self, temperatura: float, precipitacion: float) -> float:
        """
        ╔══════════════════════════════════════════════════════════════════════╗
        ║          FUNCIÓN SIMPLIFICADA PARA ALGORITMO GENÉTICO                ║
        ╚══════════════════════════════════════════════════════════════════════╝
        
        ESTE MÉTODO ES EL PUNTO DE CONEXIÓN CON EL MÓDULO 3 (Cristian)
        ════════════════════════════════════════════════════════════════
        
        Retorna SOLO el score numérico (0-100) para usar directamente
        como función de fitness en el Algoritmo Genético.
        
        
        PARÁMETROS:
        ───────────
            temperatura (float): Temperatura en °C
            precipitacion (float): Lluvia en mm
        
        
        RETORNA:
        ────────
            float: Score de amplitud (0-100)
                   Mayor score = Mejor día para sembrar
        
        
        EJEMPLO DE USO EN MÓDULO 3 (Cristian):
        ──────────────────────────────────────
        
            from fuzzy_system import SistemaDifusoSiembra
            
            # Crear instancia del sistema difuso
            sistema_difuso = SistemaDifusoSiembra()
            
            # Función de fitness para el Algoritmo Genético
            def fitness_function(individuo):
                '''
                individuo representa una fecha propuesta de siembra.
                Esta función evalúa qué tan buena es esa fecha.
                '''
                
                # Obtener predicción climática para esa fecha
                # (esto viene del Módulo 1)
                temp = obtener_temperatura_predicha(individuo.fecha)
                lluvia = obtener_lluvia_predicha(individuo.fecha)
                
                # Evaluar con sistema difuso (Módulo 2)
                fitness = sistema_difuso.obtener_score_para_fitness(temp, lluvia)
                
                return fitness  # Valor de 0 a 100
            
            
            # En el loop del GA:
            for individuo in poblacion:
                individuo.fitness = fitness_function(individuo)
        """
        
        resultado = self.evaluar(temperatura, precipitacion)
        return resultado['score_amplitud']
    
    
    def exportar_a_json(self, resultado: Dict, archivo: str | None = None) -> str:
        """
        Exporta el resultado a formato JSON.
        
        Si se proporciona archivo, guarda en disco.
        Siempre retorna el string JSON.
        """
        json_str = json.dumps(resultado, indent=2, ensure_ascii=False)
        
        if archivo:
            with open(archivo, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"✅ Resultado guardado en: {archivo}")
        
        return json_str


# ==============================================================================
#                    FUNCIÓN DE INTERFAZ SIMPLIFICADA
# ==============================================================================

# Instancia global (singleton) para evitar reinicializar el sistema
_sistema_global = None


def evaluar_dia(temperatura: float, precipitacion: float) -> Dict:
    """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              FUNCIÓN RÁPIDA PARA EVALUAR UN DÍA                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    Función de conveniencia que no requiere crear una instancia de la clase.
    
    USO:
    ────
        from fuzzy_system import evaluar_dia
        
        resultado = evaluar_dia(temperatura=27, precipitacion=10)
        print(resultado['score_amplitud'])  # 85.77
    """
    global _sistema_global
    
    if _sistema_global is None:
        _sistema_global = SistemaDifusoSiembra()
    
    return _sistema_global.evaluar(temperatura, precipitacion)


def obtener_score(temperatura: float, precipitacion: float) -> float:
    """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              FUNCIÓN ULTRA-SIMPLE PARA OBTENER SOLO EL SCORE             ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    Retorna directamente el número (0-100), ideal para el Módulo 3.
    
    USO:
    ────
        from fuzzy_system import obtener_score
        
        fitness = obtener_score(27, 10)  # Retorna: 85.77
    """
    global _sistema_global
    
    if _sistema_global is None:
        _sistema_global = SistemaDifusoSiembra()
    
    return _sistema_global.obtener_score_para_fitness(temperatura, precipitacion)



# ==========================================
# PEGAR ESTO AL FINAL DE src/fuzzy/fuzzy_system.py
# (Asegúrate que esté pegado a la izquierda, sin espacios antes del 'def')
# ==========================================

# ==========================================
# PEGAR AL FINAL DE src/fuzzy/fuzzy_system.py
# ==========================================

# ==========================================
# VERSIÓN DE DIAGNÓSTICO (PEGAR AL FINAL)
# ==========================================

# ==========================================
# PEGAR AL FINAL DE src/fuzzy/fuzzy_system.py
# ==========================================

# 1. CREAR LA INSTANCIA GLOBAL (Sacar el sistema de la caja)
# ¡¡IMPORTANTE!!: Cambia 'NombreDeTuClase' por el nombre real que viste al inicio.
# Si tu clase pide argumentos en el __init__, quizás necesites ponerlos.
# Lo normal es que sea así simple:
try:
    # Intenta adivinar nombres comunes, o pon tú el nombre correcto aquí:
    if 'SistemaDifusoSiembra' in globals():
        sistema_global = SistemaDifusoSiembra()
    elif 'FuzzySystem' in globals():
        sistema_global = FuzzySystem()
    elif 'Controlador' in globals():
        sistema_global = Controlador()
    else:
        # Si no lo encuentro, imprimimos aviso para que TÚ lo cambies manual
        print("⚠️ ATENCIÓN: Debes cambiar 'NombreDeTuClase' en la línea de abajo por el nombre real de tu class.")
        # sistema_global = NombreDeTuClase() # <--- DESCOMENTA Y EDITA ESTO SI FALLA
except Exception:
    pass

# 2. FUNCIÓN DE CONEXIÓN CORREGIDA
# ==========================================
# REEMPLAZA LA FUNCIÓN AL FINAL DE src/fuzzy/fuzzy_system.py
# ==========================================

# ==========================================
# VERSIÓN FINAL CORREGIDA (PEGAR AL FINAL)
# ==========================================

def calcular_aptitud(lluvia_val, temp_val):
    try:
        # Verificación de seguridad
        if 'sistema_global' not in globals():
            return 0.0

        # --- AQUÍ ESTABA EL ERROR ---
        # Ahora usamos los nombres reales que descubrimos: 'lluvia' y 'temperatura'
        sistema_global.simulacion.input['lluvia'] = lluvia_val       # <--- CORREGIDO
        sistema_global.simulacion.input['temperatura'] = temp_val    # <--- CORREGIDO
        
        # Calculamos
        sistema_global.simulacion.compute()
        
        # Obtenemos el resultado de forma segura
        if sistema_global.simulacion.output is None: return 0.0
        
        # Tomamos la salida (se llame 'aptitud', 'riesgo' o como sea)
        keys = list(sistema_global.simulacion.output.keys())
        if not keys: return 0.0
        
        return sistema_global.simulacion.output[keys[0]]

    except ValueError:
        # Si un dato se sale de las gráficas difusas, retornamos 0
        return 0.0
    except Exception as e:
        print(f"💀 ERROR FINAL: {e}")
        return 0.0





# ==============================================================================
#                              DEMO Y PRUEBAS
# ==============================================================================

if __name__ == "__main__":
    print()
    print("=" * 80)
    print("     MÓDULO 2: SISTEMA DE INFERENCIA DIFUSA - SIEMBRA DE MAÍZ 🌽")
    print("=" * 80)
    print()
    
    # Crear sistema
    sistema = SistemaDifusoSiembra()
    
    print()
    print("─" * 80)
    print("                         PRUEBAS DE FUNCIONAMIENTO")
    print("─" * 80)
    print()
    
    # Prueba 1: Condición óptima
    print("🧪 PRUEBA 1: Condición ÓPTIMA")
    print("   Entrada: Temp=27°C, Lluvia=10mm")
    resultado1 = sistema.evaluar(temperatura=27, precipitacion=10)
    print(f"   Salida:  Score={resultado1['score_amplitud']}, "
          f"Categoría={resultado1['categoria']}, "
          f"Recomendación={resultado1['recomendacion']}")
    print()
    
    # Prueba 2: Condición desfavorable
    print("🧪 PRUEBA 2: Condición DESFAVORABLE")
    print("   Entrada: Temp=38°C, Lluvia=2mm")
    resultado2 = sistema.evaluar(temperatura=38, precipitacion=2)
    print(f"   Salida:  Score={resultado2['score_amplitud']}, "
          f"Categoría={resultado2['categoria']}, "
          f"Recomendación={resultado2['recomendacion']}")
    print()
    
    # Prueba 3: Evaluación desde JSON (como vendría del Módulo 1)
    print("🧪 PRUEBA 3: Evaluación desde JSON (integración con Módulo 1)")
    datos_modulo1 = {
        "temperatura": 25,
        "precipitacion": 12
    }
    print(f"   Datos recibidos del Módulo 1: {datos_modulo1}")
    resultado3 = sistema.evaluar_desde_json(datos_modulo1)
    print(f"   Resultado para Módulo 3: {resultado3}")
    print()
    
    # Prueba 4: Múltiples días
    print("🧪 PRUEBA 4: Evaluación de MÚLTIPLES DÍAS")
    predicciones = [
        {"fecha": "2025-06-01", "temperatura": 25, "precipitacion": 12},
        {"fecha": "2025-06-02", "temperatura": 20, "precipitacion": 8},
        {"fecha": "2025-06-03", "temperatura": 35, "precipitacion": 30},
    ]
    print("   Predicciones del Módulo 1:")
    for p in predicciones:
        print(f"      {p}")
    
    resultados = sistema.evaluar_multiples_dias(predicciones)
    print("\n   Resultados para Módulo 3:")
    for r in resultados:
        print(f"      {r['fecha']}: Score={r['score_amplitud']}, {r['categoria']}")
    print()
    
    # Prueba 5: Score simple para GA
    print("🧪 PRUEBA 5: Score para Algoritmo Genético (Módulo 3)")
    fitness = sistema.obtener_score_para_fitness(27, 10)
    print(f"   fitness = obtener_score_para_fitness(27, 10)")
    print(f"   Resultado: {fitness}")
    print()
    
    print("=" * 80)
    print("     ✅ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
    print("=" * 80)
    print()
    print("📌 INSTRUCCIONES DE INTEGRACIÓN:")
    print()
    print("   MÓDULO 1 → MÓDULO 2:")
    print("   ─────────────────────")
    print("   datos = {'temperatura': X, 'precipitacion': Y}")
    print("   resultado = sistema.evaluar_desde_json(datos)")
    print()
    print("   MÓDULO 2 → MÓDULO 3:")
    print("   ─────────────────────")
    print("   fitness = sistema.obtener_score_para_fitness(temp, lluvia)")
    print("   # Usar 'fitness' en el Algoritmo Genético")
    print()
    print("=" * 80)
