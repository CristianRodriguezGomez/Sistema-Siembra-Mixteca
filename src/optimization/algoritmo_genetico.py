"""
Módulo de Optimización con Enjambre de Partículas (PSO).

Este script implementa el algoritmo PSO utilizando la librería Mealpy (v3)
para encontrar la fecha óptima de siembra. Maximiza una función de aptitud
basada en datos climáticos históricos y lógica difusa.
"""

import os
import sys

# ✔ PEP 8: Importaciones de terceros agrupadas
import matplotlib.pyplot as plt
import numpy as np
from mealpy import PSO, FloatVar

# --- CONFIGURACIÓN DE RUTAS ---
# Se agrega el directorio padre al path para permitir importaciones locales
# independientemente de desde dónde se ejecute el script.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# ✔ PEP 8: Importaciones locales al final
from src.neural.gestor_climatico import obtener_clima_real
from src.fuzzy.fuzzy_system import calcular_aptitud


def funcion_objetivo(solution):
    """
    Calcula la aptitud (fitness) de una solución propuesta por el PSO.

    El algoritmo PSO propone un número flotante (ej. 45.3) que representa
    el día de siembra. Esta función lo convierte a entero, recupera el clima
    para los siguientes 120 días y evalúa su viabilidad usando lógica difusa.

    Args:
        solution (list): Lista con los valores de las dimensiones (aquí solo 1).

    Returns:
        float: El puntaje total acumulado (fitness). Retorna un valor muy bajo
               (-999999) si la fecha es inválida o no hay datos (penalización).
    """
    # PSO trabaja con flotantes, convertimos a entero para representar días
    dia_siembra = int(solution[0])

    # --- 1. VALIDACIÓN DE RESTRICCIONES (PENALIZACIÓN) ---
    # Si el día está fuera del rango lógico de siembra (ej. fin de año)
    if dia_siembra < 1 or dia_siembra > 240:
        return -999999

    # --- 2. OBTENCIÓN DE DATOS ---
    datos_cultivo = obtener_clima_real(dia_siembra, duracion_cultivo=120)

    if not datos_cultivo:
        return -999999

    score_total = 0

    # --- 3. EVALUACIÓN CON LÓGICA DIFUSA ---
    for dia in datos_cultivo:
        try:
            aptitud = calcular_aptitud(dia['lluvia'], dia['temp'])
            score_total += aptitud
        except Exception:
            # ✔ PEP 8: E722 - No usar 'except' vacío.
            # Ignoramos errores puntuales de cálculo para no detener la optimización
            pass

    return score_total


def correr_optimizacion():
    """
    Configura y ejecuta la optimización por Enjambre de Partículas (PSO).

    Define el espacio de búsqueda (días 1-240), configura los hiperparámetros
    del PSO (épocas, población) y genera una gráfica de convergencia al finalizar.

    Returns:
        int: El mejor día de siembra encontrado (entero).
    """
    print("\n--- INICIANDO OPTIMIZACIÓN CON ENJAMBRE DE PARTÍCULAS (PSO) ---")
    print("Mecanismo: Mealpy Library (v3)")

    # Definimos los límites usando FloatVar (Requerido por Mealpy v3)
    limites = FloatVar(lb=[1], ub=[240], name="dia_siembra")

    # --- A. DEFINICIÓN DEL PROBLEMA ---
    problem_dict = {
        "obj_func": funcion_objetivo,
        "bounds": limites,
        "minmax": "max",      # Buscamos maximizar la aptitud
        "log_to": "console",  # Imprimir progreso en consola
    }

    # --- B. CONFIGURACIÓN DEL MODELO ---
    # epoch: Número de iteraciones (generaciones)
    # pop_size: Número de partículas (agentes) buscando simultáneamente
    model = PSO.OriginalPSO(epoch=50, pop_size=20)

    # --- C. EJECUCIÓN ---
    # solve() devuelve el mejor agente encontrado tras todas las épocas
    best_agent = model.solve(problem_dict)

    mejor_dia = int(best_agent.solution[0])
    fitness_alcanzado = best_agent.target.fitness

    # --- D. GENERACIÓN DE GRÁFICA DE CONVERGENCIA ---
    print("\n📊 Generando gráfica de convergencia...")

    # Recuperamos el historial de la mejor aptitud por época
    historia_fitness = model.history.list_global_best_fit

    plt.figure(figsize=(10, 6))
    plt.plot(historia_fitness, color='blue', linewidth=2, label="Mejor Aptitud Global")

    plt.title('Curva de Convergencia: PSO (Mealpy)', fontsize=14)
    plt.xlabel('Generaciones (Épocas)', fontsize=12)
    plt.ylabel('Aptitud (Fitness)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Guardado de la imagen
    ruta_grafica = os.path.join(parent_dir, 'convergencia_pso.png')
    plt.savefig(ruta_grafica, dpi=300)
    print(f"✅ Gráfica guardada en: {ruta_grafica}")

    # Mostrar ventana (opcional)
    plt.show()

    # --- E. REPORTE FINAL ---
    print("-" * 50)
    print(" Optimización Completada (PSO)")
    print(f" Aptitud total acumulada: {fitness_alcanzado:.2f}")
    print("-" * 50)

    return mejor_dia