# Archivo: src/neural/gestor_climatico.py
import pandas as pd
import os

# Buscamos el CSV en la carpeta correcta
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'Pronostico_2026_IA.csv')

print(f"Cargando clima desde: {CSV_PATH}")

try:
    df_clima = pd.read_csv(CSV_PATH)
    df_clima = df_clima.fillna(0)
    
    # --- CORRECCIÓN EXACTA PARA TU NUEVO CSV ---
    # Mapeamos tus columnas a los nombres internos 'temp' y 'lluvia'
    nombres_correctos = {
        'Temperatura_Predicha': 'temp', 
        'Lluvia_Predicha': 'lluvia'
    }
    df_clima = df_clima.rename(columns=nombres_correctos)
    
except Exception as e:
    print(f"ERROR LEYENDO CSV: {e}")
    df_clima = pd.DataFrame()

def obtener_clima_real(dia_inicio, duracion_cultivo=120):
    if df_clima.empty:
        return []

    idx_inicio = max(0, int(dia_inicio) - 1)
    idx_fin = idx_inicio + duracion_cultivo
    
    datos = df_clima.iloc[idx_inicio : idx_fin]
    
    # Convertimos a diccionario (rápido)
    try:
        resultado = datos[['temp', 'lluvia']].to_dict('records')
    except KeyError:
        return []
        
    return resultado