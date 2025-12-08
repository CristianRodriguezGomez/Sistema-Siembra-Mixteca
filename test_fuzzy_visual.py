import matplotlib.pyplot as plt
from src.fuzzy.fuzzy_system import sistema_global

# --- DATOS DE PRUEBA ---
# Puedes cambiar estos valores para probar diferentes climas
lluvia_prueba = 20   # mm
temp_prueba = 25     # °C

print(f"\n🔬 DIAGNÓSTICO COMPLETO DE LÓGICA DIFUSA")
print(f"    Datos de prueba: Lluvia={lluvia_prueba}mm, Temp={temp_prueba}°C\n")

# 1. Pasamos los datos ('lluvia' y 'temperatura')
sistema_global.simulacion.input['lluvia'] = lluvia_prueba
sistema_global.simulacion.input['temperatura'] = temp_prueba

# 2. Calculamos el resultado
try:
    sistema_global.simulacion.compute()
    
    # Recuperamos el valor de 'amplitud'
    resultado = sistema_global.simulacion.output['amplitud']
    print(f"✅ RESULTADO MATEMÁTICO (Defuzzificación):")
    print(f"   Score de Amplitud: {resultado:.2f} / 100")

except Exception as e:
    print(f"💀 Error de cálculo: {e}")
    exit()

# 3. IMPRIMIR LAS REGLAS (Para tu reporte)
print("\n📜 REGLAS DEL SISTEMA (Base de Conocimiento):")
print("-" * 60)
try:
    for i, regla in enumerate(sistema_global.simulacion.ctrl.rules):
        print(f"Regla #{i+1}: {regla}")
except:
    print("   (No se pudieron listar las reglas en texto)")
print("-" * 60)

# 4. GENERAR GRÁFICAS (Entradas y Salidas)
print("\n📊 Generando gráficas visuales... (Se abrirán varias ventanas)")
print("   -> Busca las líneas negras verticales que indican tus datos.")

try:
    # A) Graficar las ENTRADAS (Antecedentes)
    # Esto te mostrará en qué categoría cayeron la Lluvia y la Temperatura
    for variable in sistema_global.simulacion.ctrl.antecedents:
        variable.view(sim=sistema_global.simulacion)
    
    # B) Graficar la SALIDA (Consecuente)
    # Esto te mostrará el triángulo recortado y el resultado final
    for variable_salida in sistema_global.simulacion.ctrl.consequents:
        variable_salida.view(sim=sistema_global.simulacion)
    
    print("✅ Gráficas listas. Revisa las ventanas emergentes.")
    plt.show()
    
except Exception as e:
    print(f"⚠️ No se pudo graficar alguna variable: {e}")