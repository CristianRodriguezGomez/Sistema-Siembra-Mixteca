from src.optimization.algoritmo_genetico import correr_optimizacion

if __name__ == "__main__":
    print("--- SISTEMA DE OPTIMIZACIÓN DE SIEMBRA MIXTECA ---")
    print("Iniciando búsqueda con Algoritmos Genéticos...")
    mejor_dia = correr_optimizacion()
    print(f"Recomendación final para el agricultor: Sembrar en el día {mejor_dia} del año.")