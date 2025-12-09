"""
Script de prueba y visualización para el sistema de lógica difusa.

Este módulo permite probar el sistema difuso, ver los resultados numéricos
y guardar automáticamente las gráficas de las funciones de membresía.
"""

import os  # <--- NUEVO
import matplotlib.pyplot as plt
from src.fuzzy.fuzzy_system import sistema_global

# --- DATOS DE PRUEBA ---
lluvia_prueba = 20   # mm
temp_prueba = 25     # °C

# --- CONFIGURACIÓN DE GUARDADO (NUEVO) ---
# Creamos una carpeta para no desordenar el proyecto
current_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(current_dir, "reportes_fuzzy")
os.makedirs(output_folder, exist_ok=True) # Crea la carpeta si no existe

print(f"\n DIAGNÓSTICO COMPLETO DE LÓGICA DIFUSA")
print(f"    Datos de prueba: Lluvia={lluvia_prueba}mm, Temp={temp_prueba}°C\n")

# --- 1. Configuración de Entradas ---
sistema_global.simulacion.input['lluvia'] = lluvia_prueba
sistema_global.simulacion.input['temperatura'] = temp_prueba

# --- 2. Ejecución del Cálculo ---
try:
    sistema_global.simulacion.compute()
    resultado = sistema_global.simulacion.output['amplitud']
    print(f" RESULTADO MATEMÁTICO (Defuzzificación):")
    print(f"   Score de Amplitud: {resultado:.2f} / 100")

except Exception as e:
    print(f"Error de cálculo: {e}")
    exit()

# --- 3. Impresión de Reglas ---
print("\nREGLAS DEL SISTEMA (Base de Conocimiento):")
print("-" * 60)
try:
    for i, regla in enumerate(sistema_global.simulacion.ctrl.rules):
        print(f"Regla #{i+1}: {regla}")
except Exception:
    print("   (No se pudieron listar las reglas en texto)")
print("-" * 60)

# --- 4. Generación y Guardado de Gráficas (MODIFICADO) ---
print("\n Generando y guardando gráficas visuales...")
print(f"Las imágenes se guardarán en: {output_folder}")

try:
    # A. GRAFICAR ENTRADAS (Antecedentes)
    for variable in sistema_global.simulacion.ctrl.antecedents:
        # Esto genera la ventana visual
        variable.view(sim=sistema_global.simulacion)
        
        # Truco: Capturamos la figura activa actual
        fig = plt.gcf() 
        fig.set_size_inches(10, 5) # Hacemos la imagen un poco más ancha
        
        # Generamos el nombre del archivo basado en el nombre de la variable
        nombre_archivo = f"fuzzy_input_{variable.label}.png"
        ruta_completa = os.path.join(output_folder, nombre_archivo)
        
        # Guardamos
        fig.savefig(ruta_completa, dpi=300)
        print(f"Guardado: {nombre_archivo}")

    # B. GRAFICAR SALIDA (Consecuentes)
    for variable_salida in sistema_global.simulacion.ctrl.consequents:
        variable_salida.view(sim=sistema_global.simulacion)
        
        fig = plt.gcf()
        fig.set_size_inches(10, 5)
        
        nombre_archivo = f"fuzzy_output_{variable_salida.label}.png"
        ruta_completa = os.path.join(output_folder, nombre_archivo)
        
        fig.savefig(ruta_completa, dpi=300)
        print(f"Guardado: {nombre_archivo}")

    print("\n¡Proceso terminado! Se abrirán las ventanas para revisión final.")
    plt.show()  # Muestra las ventanas (opcional, si quieres verlas en pantalla también)

except Exception as e:
    print(f"No se pudo graficar alguna variable: {e}")