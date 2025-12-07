import pygad
import sys
import os

# Truco para importar desde la carpeta padre
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mocks import predecir_clima_simulado, evaluar_riesgo_simulado

def fitness_func(ga_instance, solution, solution_idx):
    dia_siembra = int(solution[0])
    
    # Restricción: Solo sembrar entre día 100 (abril) y 200 (julio)
    if dia_siembra < 100 or dia_siembra > 200:
        return -1000 # Penalización fuerte

    # Simulamos el ciclo de cultivo
    pronostico = predecir_clima_simulado(dia_siembra)
    riesgo_total = 0
    for dia in pronostico:
        riesgo_total += evaluar_riesgo_simulado(dia['temp'], dia['lluvia'])
    
    # Queremos MINIMIZAR el riesgo, así que el fitness es inverso al riesgo
    return 10000 - riesgo_total

def correr_optimizacion():
    ga_instance = pygad.GA(
        num_generations=50, #30
        num_parents_mating=4,
        fitness_func=fitness_func,
        sol_per_pop=20, #10
        num_genes=1,
        gene_type=int,
        init_range_low=100,
        init_range_high=200,
        #mutation_percent_genes=20
        mutation_num_genes=1
    )
    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    print(f"Mejor día para sembrar encontrado: Día {solution[0]}")
    return solution[0]