"""
Módulo Gestor Climático.

Responsable de cargar y proporcionar los datos climáticos pronosticados
para el año 2026 desde un archivo CSV.
"""

import pandas as pd
import os

# --- Carga de Datos Climáticos ---
# Construcción de la ruta al archivo CSV para asegurar que funcione en cualquier sistema.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'Pronostico_2026_IA.csv')

print(f"Cargando clima desde: {CSV_PATH}")

try:
    # Carga del archivo CSV que contiene el pronóstico del clima.
    df_clima = pd.read_csv(CSV_PATH)
    # Rellena cualquier valor faltante (NaN) con 0 para evitar errores.
    df_clima = df_clima.fillna(0)

    # Mapeo de los nombres de columna del CSV a nombres estandarizados ('temp', 'lluvia')
    # para uso interno en el sistema. Esto desacopla el código de los nombres
    # específicos del archivo CSV.
    nombres_correctos = {
        'Temperatura_Predicha': 'temp',
        'Lluvia_Predicha': 'lluvia'
    }
    df_clima = df_clima.rename(columns=nombres_correctos)

except Exception as e:
    # En caso de error al leer el archivo, se crea un DataFrame vacío
    # y se notifica al usuario para que el programa no se detenga abruptamente.
    print(f"ERROR LEYENDO CSV: {e}")
    df_clima = pd.DataFrame()


def obtener_clima_real(dia_inicio, duracion_cultivo=120):
    """
    Obtiene una porción del pronóstico climático para un ciclo de cultivo.

    Args:
        dia_inicio (int): El día del año en que comienza el ciclo (1-365).
        duracion_cultivo (int): La duración en días del ciclo de cultivo.

    Returns:
        list: Una lista de diccionarios, donde cada diccionario representa
              un día con sus valores de 'temp' y 'lluvia'. Retorna una
              lista vacía si los datos no están disponibles o hay un error.
    """
    if df_clima.empty:
        return []

    # Calcula los índices de inicio y fin para extraer los datos del DataFrame.
    idx_inicio = max(0, int(dia_inicio) - 1)
    idx_fin = idx_inicio + duracion_cultivo
    datos = df_clima.iloc[idx_inicio : idx_fin]

    try:
        # Convierte el sub-DataFrame a una lista de diccionarios para un acceso más fácil.
        resultado = datos[['temp', 'lluvia']].to_dict('records')
    except KeyError:
        # Si las columnas 'temp' o 'lluvia' no existen después del renombrado, retorna vacío.
        return []

    return resultado