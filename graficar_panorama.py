"""
Script de visualización del "Panorama de Aptitud".

Este módulo realiza un análisis de fuerza bruta para calcular el "fitness" o
aptitud de siembra para cada día posible del año. Luego, genera una gráfica
que muestra cómo varía esta aptitud, destacando el día óptimo.

Sirve como una herramienta de validación y análisis para entender el
comportamiento del sistema y comparar el resultado del algoritmo genético
con el "óptimo real" encontrado por fuerza bruta.
"""

import matplotlib.pyplot as plt
from src.neural.gestor_climatico import obtener_clima_real
from src.fuzzy.fuzzy_system import calcular_aptitud

# Configuración estética
plt.style.use('ggplot')


def obtener_score_del_dia(dia_inicio):
    """
    Calcula el puntaje de aptitud total para un ciclo de cultivo que inicia en un día específico.

    Args:
        dia_inicio (int): El día del año para el cual se calculará el score.

    Returns:
        float: El puntaje de aptitud acumulado para todo el ciclo de cultivo.
               Retorna 0 si no hay datos climáticos disponibles.
    """
    datos = obtener_clima_real(dia_inicio, duracion_cultivo=120)

    if not datos:
        return 0

    score = 0
    for dia in datos:
        try:
            # Para cada día del ciclo, calcula su aptitud usando el sistema difuso
            # y la suma al puntaje total.
            s = calcular_aptitud(dia['lluvia'], dia['temp'])
            score += s
        except Exception:
            # Si hay un error en el cálculo para un día, simplemente se omite.
            pass
    return score


print("⏳ Generando el Panorama Completo (Esto puede tardar unos segundos)...")

# --- 1. Análisis de Fuerza Bruta ---
# Se itera sobre cada día posible de inicio de siembra (limitado a 240 para asegurar un ciclo completo).
dias = list(range(1, 241))
scores = []

for d in dias:
    if d % 20 == 0:
        print(f"   ... Calculando día {d}")
    scores.append(obtener_score_del_dia(d))

# --- 2. Identificación del Óptimo Real ---
# Se encuentra el puntaje máximo y el día correspondiente en los resultados de fuerza bruta.
max_score = max(scores)
mejor_dia_real = dias[scores.index(max_score)]

# --- 3. Generación de la Gráfica ---
plt.figure(figsize=(12, 6))

plt.plot(dias, scores, color='#2ecc71', linewidth=2, label='Aptitud (Score)')
plt.fill_between(dias, scores, color='#2ecc71', alpha=0.3)

# Se destaca el punto óptimo encontrado con un marcador y una anotación.
plt.plot(mejor_dia_real, max_score, 'ro', markersize=10, label=f'Óptimo Real (Día {mejor_dia_real})')
plt.annotate(f'Mejor Fecha\nDía {mejor_dia_real}',
             xy=(mejor_dia_real, max_score),
             xytext=(mejor_dia_real+10, max_score),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('Panorama de Aptitud de Siembra (2026)', fontsize=16)
plt.xlabel('Día de Inicio de Siembra (Día del Año)', fontsize=12)
plt.ylabel('Puntaje Total (Fitness)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Etiquetas de meses aproximados en el eje X
posiciones_meses = [1, 32, 60, 91, 121, 152, 182, 213, 244]
nombres_meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep']
plt.xticks(posiciones_meses, nombres_meses)

plt.tight_layout()
print(f"¡Listo! El mejor día calculado a fuerza bruta fue: {mejor_dia_real}")
plt.show()