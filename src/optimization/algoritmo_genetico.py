import pygad
import sys
import os

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Esto permite que Python encuentre las carpetas 'src', 'neural' y 'fuzzy'
# sin importar desde dónde ejecutes el código.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir)) # Va a la raíz del proyecto
sys.path.append(parent_dir)

# --- 2. IMPORTACIONES REALES (YA NO USAMOS MOCKS) ---
# Traemos el lector del CSV de tu compañera (Fase 1)
from src.neural.gestor_climatico import obtener_clima_real

# Traemos el evaluador difuso de tu compañero (Fase 2)
# NOTA: Asegúrate que la función en fuzzy_system.py se llame 'calcular_aptitud'
# Si se llama diferente, cambia el nombre aquí abajo.
from src.fuzzy.fuzzy_system import calcular_aptitud

def fitness_func(ga_instance, solution, solution_idx):
    dia_siembra = int(solution[0])
    
    # --- 1. PRIMERO LAS RESTRICCIONES (Vallas de Seguridad) ---
    # Si el día no sirve, lo rechazamos INMEDIATAMENTE.
    # Con archivo anual: Límite 240 (Agosto) para cosechar en Diciembre.
    if dia_siembra < 1 or dia_siembra > 240:
        return -9999

    # --- 2. AHORA SÍ, PEDIMOS DATOS ---
    datos_cultivo = obtener_clima_real(dia_siembra, duracion_cultivo=120)

    # Protección extra: Si por alguna razón no hay datos (lista vacía)
    if not datos_cultivo:
        return -9999

    score_total = 0

    # --- 3. EVALUAR CADA DÍA (Ciclo de Cultivo) ---
    for dia in datos_cultivo:
        temp = dia['temp']
        lluvia = dia['lluvia']

        try:
            # Llamamos a Fuzzy Logic (asegúrate que el orden sea Lluvia, Temp)
            aptitud_dia = calcular_aptitud(lluvia, temp)
            score_total += aptitud_dia
        except Exception:
            pass 

    return score_total

def correr_optimizacion():
    ga_instance = pygad.GA(
        num_generations=50,       # Puedes subirlo a 100 si quieres más precisión
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=20,           # Puedes subirlo a 30 o 40
        num_genes=1,
        gene_type=int,
        
        # --- AQUÍ ESTÁ EL CAMBIO IMPORTANTE ---
        # Rango inicial: Que pruebe fechas desde Enero (1) hasta Agosto (240)
        init_range_low=1,
        init_range_high=240,
        
        # Espacio permitido: Nunca salirse de este rango
        gene_space={'low': 1, 'high': 240},
        
        mutation_num_genes=1,
        
        # Si quieres resultados variados en cada prueba, borra o comenta el random_seed
        # random_seed=42 
    )
    print("🌱 Iniciando evolución con DATOS REALES...")
    ga_instance.run()
    ga_instance.plot_fitness()
    solution, solution_fitness, _ = ga_instance.best_solution()
    mejor_dia = int(solution[0])
    
    print(f"--------------------------------------------------")
    print(f"✅ Optimización Completada")
    print(f"📅 Mejor fecha de inicio sugerida: Día {mejor_dia} del año")
    print(f"🏆 Aptitud (Fitness) alcanzada: {solution_fitness:.2f}")
    print(f"--------------------------------------------------")
    
    return mejor_dia