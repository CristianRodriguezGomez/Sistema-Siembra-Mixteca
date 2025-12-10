# Archivo: src/optimization/algoritmo_genetico.py
import pygad
import numpy as np
import os
import sys

# Agregamos la ruta del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.neural.gestor_climatico import obtener_clima_real
from src.fuzzy.fuzzy_system import calcular_aptitud

def fitness_func(ga_instance, solution, solution_idx):
    """
    Función de Aptitud (Fitness) para el Algoritmo Genético.
    Evalúa qué tan buena es una fecha de siembra.
    """
    dia_siembra = int(solution[0])

    # 1. RESTRICCIONES (Vallas de Seguridad)
    # Límite 240 para asegurar cosecha antes de fin de año
    if dia_siembra < 1 or dia_siembra > 240:
        return -999999

    # 2. OBTENER DATOS DEL CLIMA
    datos_cultivo = obtener_clima_real(dia_siembra, duracion_cultivo=120)

    if not datos_cultivo:
        return -999999

    score_total = 0

    # 3. EVALUAR CON LÓGICA DIFUSA
    for dia in datos_cultivo:
        try:
            # Sumamos la aptitud de cada día del ciclo
            aptitud = calcular_aptitud(dia['lluvia'], dia['temp'])
            score_total += aptitud
        except Exception:
            pass

    return score_total

def correr_optimizacion():
    print("\n--- INICIANDO ALGORITMO GENÉTICO (PyGAD) ---")
    
    # Configuración del AG
    ga_instance = pygad.GA(
        num_generations=50,       # Número de generaciones
        num_parents_mating=5,     # Padres para cruza
        fitness_func=fitness_func,
        sol_per_pop=20,           # Individuos por población
        num_genes=1,              # Solo buscamos 1 variable (el día)
        gene_type=int,            # Tiene que ser un día entero
        
        # Rango de búsqueda (Enero a Agosto)
        init_range_low=1,
        init_range_high=240,
        
        # Restricción estricta (Gen Space)
        gene_space={'low': 1, 'high': 240},
        
        mutation_num_genes=1,
        # random_seed=42  # Descomenta si quieres resultados fijos
    )

    # Ejecutar
    ga_instance.run()

    # Resultados
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    mejor_dia = int(solution[0])
    
    print("-" * 50)
    print(" Optimización Completada (Algoritmo Genético)")
    print(f" Mejor día encontrado: {mejor_dia}")
    print(f" Aptitud alcanzada: {solution_fitness:.2f}")
    print("-" * 50)

    # Generar gráfica de evolución
    print("Generando gráfica de evolución genética...")
    ga_instance.plot_fitness()
    
    return mejor_dia