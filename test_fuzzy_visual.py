"""
Script de prueba y visualización para el sistema de lógica difusa.

Este módulo permite probar el sistema difuso con valores específicos de
lluvia y temperatura. Muestra el resultado numérico (defuzzificación),
imprime las reglas activadas y genera gráficas detalladas de las funciones
de membresía de entrada y salida. Es una herramienta de diagnóstico clave
para entender y depurar el comportamiento del sistema difuso.
"""

import matplotlib.pyplot as plt
from src.fuzzy.fuzzy_system import sistema_global

# --- DATOS DE PRUEBA ---
# Estos valores pueden ser modificados para probar diferentes escenarios climáticos.
lluvia_prueba = 20   # mm
temp_prueba = 25     # °C

print(f"\n DIAGNÓSTICO COMPLETO DE LÓGICA DIFUSA")
print(f"    Datos de prueba: Lluvia={lluvia_prueba}mm, Temp={temp_prueba}°C\n")

# --- 1. Configuración de las Entradas de la Simulación ---
# Se asignan los valores de prueba a las variables de entrada del sistema difuso.
sistema_global.simulacion.input['lluvia'] = lluvia_prueba
sistema_global.simulacion.input['temperatura'] = temp_prueba

# --- 2. Ejecución del Cálculo Difuso ---
try:
    # El método compute() ejecuta todo el proceso: fuzzificación, inferencia y defuzzificación.
    sistema_global.simulacion.compute()

    # Se recupera el valor de salida ('amplitud') después del cálculo.
    resultado = sistema_global.simulacion.output['amplitud']
    print(f" RESULTADO MATEMÁTICO (Defuzzificación):")
    print(f"   Score de Amplitud: {resultado:.2f} / 100")

except Exception as e:
    print(f"Error de cálculo: {e}")
    exit()  # Termina el script si el cálculo falla.

# --- 3. Impresión de la Base de Conocimiento (Reglas) ---
print("\nREGLAS DEL SISTEMA (Base de Conocimiento):")
print("-" * 60)
try:
    # Itera sobre todas las reglas definidas en el controlador difuso y las imprime.
    for i, regla in enumerate(sistema_global.simulacion.ctrl.rules):
        print(f"Regla #{i+1}: {regla}")
except Exception:
    print("   (No se pudieron listar las reglas en texto)")
print("-" * 60)

# --- 4. Generación de Gráficas de Diagnóstico ---
print("\n Generando gráficas visuales... (Se abrirán varias ventanas)")
print("   -> Busca las líneas negras verticales que indican tus datos.")

try:
    # Grafica las funciones de membresía para cada variable de entrada (antecedentes).
    # La línea vertical negra indica el valor de entrada de prueba.
    for variable in sistema_global.simulacion.ctrl.antecedents:
        variable.view(sim=sistema_global.simulacion)

    # Grafica la función de membresía de la variable de salida (consecuente).
    # Muestra el área agregada y la línea vertical del resultado defuzzificado.
    for variable_salida in sistema_global.simulacion.ctrl.consequents:
        variable_salida.view(sim=sistema_global.simulacion)

    print("Gráficas listas. Revisa las ventanas emergentes.")
    plt.show()  # Muestra todas las ventanas de gráficos generadas.

except Exception as e:
    print(f"⚠️ No se pudo graficar alguna variable: {e}")