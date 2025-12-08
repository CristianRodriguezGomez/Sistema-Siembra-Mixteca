"""
Punto de entrada principal para el Sistema de Optimización de Siembra Mixteca.

Este script ejecuta el algoritmo genético para encontrar el mejor día del año
para la siembra y presenta el resultado al usuario en un formato de fecha claro.
"""

import datetime

from src.optimization.algoritmo_genetico import correr_optimizacion

if __name__ == "__main__":
    # --- 1. Ejecución del Algoritmo de Optimización ---
    print("--- SISTEMA DE OPTIMIZACIÓN DE SIEMBRA MIXTECA ---")
    print("Iniciando búsqueda con Algoritmos Genéticos...")

    # Llama a la función principal que contiene la lógica del algoritmo genético.
    mejor_dia = correr_optimizacion()

    print(f"Recomendación final para el agricultor: Sembrar en el día {mejor_dia} del año.")

# --- 2. Conversión del resultado a una fecha legible ---
# Se asume que el pronóstico y la recomendación son para el año 2026.
fecha_inicio = datetime.date(2026, 1, 1)
fecha_final = fecha_inicio + datetime.timedelta(days=int(mejor_dia) - 1)

# Diccionario para traducir el número del mes a su nombre en español.
meses_es = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

dia = fecha_final.day
mes = meses_es[fecha_final.month]
anio = fecha_final.year

# --- 3. Presentación del resultado final ---
print(f"Fecha exacta recomendada: {dia} de {mes} de {anio}")