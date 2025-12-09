"""
Punto de entrada principal para el Sistema de Optimización de Siembra Mixteca.

Este script ejecuta el algoritmo de optimización (PSO/Genético) y, además de dar
la fecha, genera una gráfica del pronóstico climático para el ciclo de cultivo seleccionado.
"""

import datetime
import matplotlib.pyplot as plt
import os
import sys

# --- IMPORTACIONES ---
from src.optimization.algoritmo_genetico import correr_optimizacion
# Necesitamos esta función para graficar el clima del periodo ganador
from src.neural.gestor_climatico import obtener_clima_real 

if __name__ == "__main__":
    # --- 1. Ejecución del Algoritmo de Optimización ---
    print("--- SISTEMA DE OPTIMIZACIÓN DE SIEMBRA MIXTECA ---")
    print("Iniciando búsqueda de la mejor ventana de siembra...")

    # Obtiene el día (número entero 1-365)
    mejor_dia = correr_optimizacion()

    print(f"Recomendación final para el agricultor: Sembrar en el día {mejor_dia} del año.")

    # --- 2. Conversión del resultado a una fecha legible ---
    fecha_inicio = datetime.date(2026, 1, 1) + datetime.timedelta(days=int(mejor_dia) - 1)
    
    # Calculamos también la fecha de cosecha (120 días después)
    fecha_cosecha = fecha_inicio + datetime.timedelta(days=120)

    meses_es = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    print(f"\nFecha exacta recomendada: {fecha_inicio.day} de {meses_es[fecha_inicio.month]} de {fecha_inicio.year}")
    print(f"Fecha estimada de cosecha: {fecha_cosecha.day} de {meses_es[fecha_cosecha.month]} de {fecha_cosecha.year}")

    # --- 3. GENERACIÓN DE GRÁFICA DE LA VENTANA SELECCIONADA (NUEVO) ---
    print("\nGenerando gráfica del clima para el periodo seleccionado...")

    # Recuperamos los datos climáticos SOLO de la ventana ganadora (120 días)
    datos_cultivo = obtener_clima_real(mejor_dia, duracion_cultivo=120)
    
    if datos_cultivo:
        # Extraemos listas para graficar
        dias_ciclo = list(range(1, 121))
        temps = [d['temp'] for d in datos_cultivo]
        lluvias = [d['lluvia'] for d in datos_cultivo]

        # Configuración de la figura con DOS ejes Y (Doble escala)
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # EJE Y1 (Izquierda): Temperatura
        color_temp = 'tab:red'
        ax1.set_xlabel('Días del Ciclo de Cultivo (1-120)')
        ax1.set_ylabel('Temperatura (°C)', color=color_temp, fontsize=12)
        ax1.plot(dias_ciclo, temps, color=color_temp, linewidth=2, label='Temperatura')
        ax1.tick_params(axis='y', labelcolor=color_temp)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # EJE Y2 (Derecha): Lluvia
        ax2 = ax1.twinx()  # Instancia un segundo eje que comparte el mismo eje X
        color_rain = 'tab:blue'
        ax2.set_ylabel('Lluvia (mm)', color=color_rain, fontsize=12)
        # Usamos gráfico de barras para la lluvia (se ve mejor) o fill_between
        ax2.fill_between(dias_ciclo, lluvias, color=color_rain, alpha=0.3, label='Lluvia Acumulada')
        ax2.tick_params(axis='y', labelcolor=color_rain)

        # Títulos y Leyendas
        plt.title(f'Condiciones Climáticas para la Ventana de Siembra: Día {mejor_dia}', fontsize=14)
        
        # Ajuste para guardar
        fig.tight_layout()
        
        nombre_archivo = 'resultado_ventana_seleccionada.png'
        # Guardar en la raíz o donde desees
        ruta_guardado = os.path.join(os.getcwd(), nombre_archivo)
        plt.savefig(ruta_guardado, dpi=300)
        
        print(f" Gráfica del ciclo guardada en: {ruta_guardado}")
        plt.show()
    else:
        print("No se pudieron recuperar datos climáticos para graficar.")