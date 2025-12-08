import datetime

from src.optimization.algoritmo_genetico import correr_optimizacion

if __name__ == "__main__":
    print("--- SISTEMA DE OPTIMIZACIÓN DE SIEMBRA MIXTECA ---")
    print("Iniciando búsqueda con Algoritmos Genéticos...")
    mejor_dia = correr_optimizacion()
    print(f"Recomendación final para el agricultor: Sembrar en el día {mejor_dia} del año.")

# Convertimos a fecha
fecha_inicio = datetime.date(2026, 1, 1)
fecha_final = fecha_inicio + datetime.timedelta(days=int(mejor_dia) - 1)

# Truco para imprimir el mes en español sin configurar locales complejos
meses_es = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

dia = fecha_final.day
mes = meses_es[fecha_final.month]
anio = fecha_final.year

print(f"📅 Fecha exacta recomendada: {dia} de {mes} de {anio}")