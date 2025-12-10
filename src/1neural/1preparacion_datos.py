"""
Este código de Python realiza la preparación completa de datos climáticos históricos para 
la primera fase del proyecto de Cómputo Flexible (Predicción con LSTM). 
El script automatiza la descarga, limpieza, ingeniería de características (codificación cíclica) 
y la reestructuración de la serie de tiempo en el formato de ventanas 
deslizantes (Sliding Window) necesario para entrenar una Red Neuronal Recurrente.

"""

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURACIÓN (Aquí se agregan las coordenadas deseadas Huajuapan) ---
LAT = 17.8058   # Coordenadas de Huajuapan de León
LON = -97.7784
START_DATE = "20050101" # Desde el 1 de Enero de 2005
END_DATE = "20240101"   # Hasta el 1 de Enero de 2024
VENTANA_DIAS = 15       # Cuántos días atrás mirará la IA para predecir

# --- PASO 1: EL MENSAJERO ---
def descargar_datos():
    print("1. Conectando con satélites de NASA POWER...")
    
    # URL base y parámetros
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = "T2M,PRECTOTCORR"    
    # Construimos la URL
    url = f"{base_url}?parameters={params}&community=AG&longitude={LON}&latitude={LAT}&start={START_DATE}&end={END_DATE}&format=JSON"
    
    print(f"   (Consultando: {url})") # Imprimimos la URL para verificarla
    
    try:
        # Aumentamos el timeout a 30 segundos porque 20 años de datos pesan
        response = requests.get(url, timeout=30) 
        
        if response.status_code == 200:
            data = response.json()
            
            # Verificamos si la NASA devolvió datos vacíos
            if 'properties' not in data:
                print(" La NASA respondió, pero sin datos válidos.")
                print(data)
                return None

            df = pd.DataFrame({
                'Fecha': pd.to_datetime(list(data['properties']['parameter']['T2M'].keys()), format='%Y%m%d'),
                'Temperatura': list(data['properties']['parameter']['T2M'].values()),
                'Lluvia': list(data['properties']['parameter']['PRECTOTCORR'].values())
            })
            
            df['Dia_Anio'] = df['Fecha'].dt.dayofyear
            print(f"Descarga lista: {len(df)} días de historia obtenidos.")
            return df
            
        else:
            # AQUÍ ESTÁ EL CAMBIO IMPORTANTE: Nos dirá qué error es
            print(f"Error de conexión. Código: {response.status_code}")
            print(f"   Mensaje de NASA: {response.text}")
            return None
            
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None

# --- PASO 2: EL LIMPIADOR Y TRADUCTOR (Procesamiento) ---
def procesar_datos(df):
    print("2. Limpiando y Normalizando...")
    
    # A. Limpieza: Si hay un -999 (error), se pone el valor del día anterior
    df = df.replace(-999, np.nan).ffill().bfill()

    # B. Matemáticas Cíclicas (El truco para la siembra)
    # Esto le enseña a la IA que el 31 de Dic está pegado al 1 de Enero
    df['Dia_Sin'] = np.sin(2 * np.pi * df['Dia_Anio'] / 365.0)
    df['Dia_Cos'] = np.cos(2 * np.pi * df['Dia_Anio'] / 365.0)

    # C. Normalización (Escalar todo entre 0 y 1 para la Red Neuronal)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[['Temp_Norm', 'Lluvia_Norm']] = scaler.fit_transform(df[['Temperatura', 'Lluvia']])
    
    return df

# --- PASO 3: EL ARQUITECTO (Crear ventanas para la IA) ---
def crear_dataset_ia(df, ventana):
    print(f"3. Construyendo ventanas de {ventana} días...")
    datos_x = []
    
    # Usamos: Temperatura, Lluvia y la Fecha (Seno/Coseno) como entrada
    columnas_input = ['Temp_Norm', 'Lluvia_Norm', 'Dia_Sin', 'Dia_Cos']
    vals_input = df[columnas_input].values
    vals_target = df[['Temp_Norm', 'Lluvia_Norm']].values # Lo que queremos predecir
    
    for i in range(len(vals_input) - ventana):
        # Tomamos 'ventana' días del pasado
        historia = vals_input[i:i+ventana].flatten()
        # Tomamos el día siguiente como la respuesta correcta
        objetivo = vals_target[i+ventana]
        
        datos_x.append(list(historia) + list(objetivo))
        
    # Ponemos nombres a las columnas (para que no se pierdan)
    cols = []
    for d in range(ventana):
        cols.append(f'Temp_Dia_{d+1}')
        cols.append(f'Lluvia_Dia_{d+1}')
        cols.append(f'Fecha_Sin_{d+1}')
        cols.append(f'Fecha_Cos_{d+1}')
    cols.append('TARGET_Temp_Manana')
    cols.append('TARGET_Lluvia_Manana')
    
    return pd.DataFrame(datos_x, columns=cols)

# --- EJECUCIÓN ------
if __name__ == "__main__":
    # 1. Ejecutar extracción
    datos = descargar_datos()
    
    if datos is not None:
        # 2. Ejecutar limpieza
        datos_listos = procesar_datos(datos)
        
        # Guardamos un archivo "Humano" para que tú lo revises en Excel
        datos_listos[['Fecha', 'Temperatura', 'Lluvia', 'Dia_Anio']].to_csv('Reporte_Humano_Huajuapan.csv', index=False)
        print(" Archivo 'Reporte_Humano_Huajuapan.csv' creado (para ver en Excel).")

        # 3. Crear formato para la Red Neuronal
        dataset_ia = crear_dataset_ia(datos_listos, ventana=VENTANA_DIAS)
        
        # Guardamos el archivo "Máquina" para tus compañeros
        dataset_ia.to_csv('Dataset_Entrenamiento_IA.csv', index=False)
        print(f" Archivo 'Dataset_Entrenamiento_IA.csv' creado con éxito.")
        print(f"   -> Tiene {len(dataset_ia)} ejemplos de entrenamiento.")
        print(f"   -> Cada ejemplo usa {VENTANA_DIAS} días de historia para predecir el siguiente.")
