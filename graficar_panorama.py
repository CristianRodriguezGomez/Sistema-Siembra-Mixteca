import matplotlib.pyplot as plt
import numpy as np
from src.neural.gestor_climatico import obtener_clima_real
from src.fuzzy.fuzzy_system import calcular_aptitud

# Configuración estética
plt.style.use('ggplot')

def obtener_score_del_dia(dia_inicio):
    """Calcula el fitness real para un día específico sin el GA"""
    datos = obtener_clima_real(dia_inicio, duracion_cultivo=120)
    
    if not datos: return 0
    
    score = 0
    for dia in datos:
        try:
            # Sumamos la calidad de cada uno de los 120 días
            s = calcular_aptitud(dia['lluvia'], dia['temp'])
            score += s
        except:
            pass
    return score

print("⏳ Generando el Panorama Completo (Esto puede tardar unos segundos)...")

# 1. ESCANEAMOS TODO EL AÑO (Del día 1 al 240)
dias = list(range(1, 241))
scores = []

for d in dias:
    # Imprimimos progreso cada 20 días
    if d % 20 == 0: print(f"   ... Calculando día {d}")
    scores.append(obtener_score_del_dia(d))

# 2. ENCONTRAMOS EL MEJOR DÍA MANUALMENTE (Para comparar)
max_score = max(scores)
mejor_dia_real = dias[scores.index(max_score)]

# 3. GRAFICAMOS
plt.figure(figsize=(12, 6))

# Dibujamos la línea de aptitud
plt.plot(dias, scores, color='#2ecc71', linewidth=2, label='Aptitud (Score)')

# Rellenamos el área bajo la curva
plt.fill_between(dias, scores, color='#2ecc71', alpha=0.3)

# Marcamos el punto máximo encontrado
plt.plot(mejor_dia_real, max_score, 'ro', markersize=10, label=f'Óptimo Real (Día {mejor_dia_real})')
plt.annotate(f'Mejor Fecha\nDía {mejor_dia_real}', 
             xy=(mejor_dia_real, max_score), 
             xytext=(mejor_dia_real+10, max_score),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Decoración
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
print(f"✅ ¡Listo! El mejor día calculado a fuerza bruta fue: {mejor_dia_real}")
plt.show()