import pygad
import sys
import os

# ✔ PEP8: Importaciones estándar → terceros → locales (este orden está bien)
# ✔ PEP8: líneas en blanco correctas entre bloques


# --- 1. CONFIGURACIÓN DE RUTAS ---
# Esto permite que Python encuentre las carpetas 'src', 'neural' y 'fuzzy'
# sin importar desde dónde ejecutes el código.

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # ✔ PEP8: espacio antes de comentario inline
sys.path.append(parent_dir)

# ✔ PEP8: Nombres de variables en snake_case correctos
# ✔ PEP8: Líneas <= 79-99 caracteres recomendadas (estas líneas están bien)


# --- 2. IMPORTACIONES REALES (YA NO USAMOS MOCKS) ---
# Traemos el lector del CSV de tu compañera (Fase 1)
from src.neural.gestor_climatico import obtener_clima_real  # ✔ Importación específica OK

# Traemos el evaluador difuso (Fase 2)
# ✔ PEP8: Comentarios útiles y concisos
from src.fuzzy.fuzzy_system import calcular_aptitud


def fitness_func(ga_instance, solution, solution_idx):
    # ✔ Nombre de función en snake_case
    # ✔ Argumentos descriptivos
    dia_siembra = int(solution[0])

    # --- 1. RESTRICCIONES ---
    if dia_siembra < 1 or dia_siembra > 240:
        return -9999  # ✔ Constante mágica está bien aquí como penalización

    # --- 2. OBTENER DATOS ---
    datos_cultivo = obtener_clima_real(dia_siembra, duracion_cultivo=120)

    # ✔ PEP8: Comparar con truthiness es válido ("if not datos:")
    if not datos_cultivo:
        return -9999

    score_total = 0  # ✔ Nombre en snake_case, evita abreviaturas opacas

    # --- 3. EVALUACIÓN DÍA A DÍA ---
    for dia in datos_cultivo:
        temp = dia['temp']
        lluvia = dia['lluvia']

        try:
            aptitud_dia = calcular_aptitud(lluvia, temp)
            score_total += aptitud_dia
        except Exception:  # ⚠ PEP8: Capturar Exception genérico no es recomendado.
            # Sugerencia: capturar excepciones específicas.
            pass

    return score_total


def correr_optimizacion():
    # ✔ Nombre en snake_case
    ga_instance = pygad.GA(
        num_generations=50,        # ✔ Comentarios concisos
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=20,
        num_genes=1,
        gene_type=int,

        # --- RANGO INICIAL ---
        init_range_low=1,
        init_range_high=240,

        # ✔ Diccionario con min/max — bien formateado
        gene_space={'low': 1, 'high': 240},

        mutation_num_genes=1,
        # random_seed=42  # ✔ Comentado correctamente
    )

    print("Iniciando evolución con DATOS REALES...")
    ga_instance.run()
    ga_instance.plot_fitness()

    solution, solution_fitness, _ = ga_instance.best_solution()
    mejor_dia = int(solution[0])  # ✔ Convierte explicitamente a int

    print(f"--------------------------------------------------")
    print(f" Optimización Completada")
    print(f" Mejor fecha de inicio sugerida: Día {mejor_dia} del año")
    print(f" Aptitud (Fitness) alcanzada: {solution_fitness:.2f}")
    print(f"--------------------------------------------------")

    return mejor_dia
