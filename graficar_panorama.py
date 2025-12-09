"""
Script de visualización del "Panorama de Aptitud".

Este módulo realiza un análisis de fuerza bruta para calcular el "fitness" o
aptitud de siembra para cada día posible del año. Luego, genera una gráfica
que muestra cómo varía esta aptitud, destacando el día óptimo.
"""

import os  # <--- NUEVO: Para manejar rutas de archivos
import matplotlib.pyplot as plt
from src.neural.gestor_climatico import obtener_clima_real
from src.fuzzy.fuzzy_system import calcular_aptitud

# Configuración estética
plt.style.use('ggplot')

# --- CONFIGURACIÓN DE RUTAS (NUEVO) ---
# Calculamos la ruta base del proyecto para guardar la imagen ahí
current_dir = os.path.dirname(os.path.abspath(__file__))
# Asumimos que el script está en una subcarpeta (ej: src/viz), subimos al root
project_root = os.path.dirname(os.path.dirname(current_dir)) 

def obtener_score_del_dia(dia_inicio):
    """
    Calcula el puntaje de aptitud total para un ciclo de cultivo.
    """
    datos = obtener_clima_real(dia_inicio, duracion_cultivo=120)

    if not datos:
        return 0

    score = 0
    for dia in datos:
        try:
            s = calcular_aptitud(dia['lluvia'], dia['temp'])
            score += s
        except Exception:
            pass
    return score


print("⏳ Generando el Panorama Completo (Esto puede tardar unos segundos)...")

# --- 1. Análisis de Fuerza Bruta ---
dias = list(range(1, 241))
scores = []

for d in dias:
    if d % 20 == 0:
        print(f"   ... Calculando día {d}")
    scores.append(obtener_score_del_dia(d))

# --- 2. Identificación del Óptimo Real ---
max_score = max(scores)
mejor_dia_real = dias[scores.index(max_score)]

# --- 3. Generación de la Gráfica ---
plt.figure(figsize=(12, 6))

plt.plot(dias, scores, color='#2ecc71', linewidth=2, label='Aptitud (Score)')
plt.fill_between(dias, scores, color='#2ecc71', alpha=0.3)

# Se destaca el punto óptimo encontrado
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

# Etiquetas de meses
posiciones_meses = [1, 32, 60, 91, 121, 152, 182, 213, 244]
nombres_meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep']
plt.xticks(posiciones_meses, nombres_meses)

plt.tight_layout()

print(f"¡Listo! El mejor día calculado a fuerza bruta fue: {mejor_dia_real}")

# --- 4. GUARDADO DE IMAGEN (NUEVO) ---
# Guardamos la imagen en la raíz del proyecto (o donde prefieras)
nombre_archivo = 'panorama_aptitud_real.png'
ruta_guardado = os.path.join(os.getcwd(), nombre_archivo) # Guarda donde ejecutes el script

# dpi=300 asegura alta resolución para documentos
plt.savefig(ruta_guardado, dpi=300) 
print(f"✅ Gráfica guardada exitosamente en: {ruta_guardado}")

plt.show()