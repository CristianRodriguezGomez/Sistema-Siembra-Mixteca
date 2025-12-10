"""
Este código implementa la parte de Generación del Pronóstico (generar_pronostico.py), 
correspondiente a la etapa final de la Fase 1. 
Su función es tomar el modelo LSTM entrenado (mejor_modelo_clima.h5) y usarlo de manera recursiva 
para generar una secuencia larga de predicciones (multi-step prediction) para el año 2026.
"""


import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta, date
from preparacion_datos import VENTANA_DIAS

# CONFIGURACIÓN DE LA PREDICCIÓN 
DIAS_A_PREDECIR = 1094 
N_FEATURES = 4 

#HERRAMIENTAS DE PREPARACIÓN

def cargar_datos_historicos_y_escalador():
    print("Cargando datos históricos e inicializando Scaler...")
    try:
        df_historico = pd.read_csv('Reporte_Humano_Huajuapan.csv')
    except FileNotFoundError:
        print("ERROR: No se encontró 'Reporte_Humano_Huajuapan.csv'. Asegúrate de ejecutar preparacion_datos.py.")
        return None, None, None
        
    scaler = MinMaxScaler(feature_range=(0, 1))
    datos_para_scaler = df_historico[['Temperatura', 'Lluvia']].values
    scaler.fit(datos_para_scaler)
    
    df_inicio = df_historico.tail(VENTANA_DIAS).copy() 
    
    df_inicio[['Temp_Norm', 'Lluvia_Norm']] = scaler.transform(df_inicio[['Temperatura', 'Lluvia']])
    
    df_inicio['Fecha'] = pd.to_datetime(df_inicio['Fecha'])
    df_inicio['Dia_Anio'] = df_inicio['Fecha'].dt.dayofyear
    df_inicio['Dia_Sin'] = np.sin(2 * np.pi * df_inicio['Dia_Anio'] / 365.0)
    df_inicio['Dia_Cos'] = np.cos(2 * np.pi * df_inicio['Dia_Anio'] / 365.0) 
    
    input_seq = df_inicio[['Temp_Norm', 'Lluvia_Norm', 'Dia_Sin', 'Dia_Cos']].values
    input_seq = input_seq.reshape(1, VENTANA_DIAS, N_FEATURES)
    return input_seq, scaler, df_inicio['Fecha'].iloc[-1] 

#PREDICCIÓN RECURSIVA

def predecir_recursivamente(input_seq_inicial, scaler, ultima_fecha_historica):
    print(f"Generando pronóstico recursivo para {DIAS_A_PREDECIR} días...")
    
    try:
        modelo = load_model('mejor_modelo_clima.h5')
    except OSError:
        print("ERROR: No se encontró 'mejor_modelo_clima.h5'. ¡Asegúrate de entrenar el modelo primero!")
        return None, None
        
    current_input = input_seq_inicial.copy()
    pronosticos_normalizados = []
    fechas_pronostico = []
    
    for i in range(DIAS_A_PREDECIR):
        prediccion_norm = modelo.predict(current_input, verbose=0)[0]
        pronosticos_normalizados.append(prediccion_norm)
        
        fecha_predicha = ultima_fecha_historica + timedelta(days=i + 1)
        fechas_pronostico.append(fecha_predicha)
        
        dia_anual = fecha_predicha.timetuple().tm_yday
        dia_sin = np.sin(2 * np.pi * dia_anual / 365.0)
        dia_cos = np.cos(2 * np.pi * dia_anual / 365.0) 
        
        nuevo_input_dia = np.array([prediccion_norm[0], prediccion_norm[1], dia_sin, dia_cos])
        current_input = np.delete(current_input[0], 0, axis=0)
        current_input = np.append(current_input, [nuevo_input_dia], axis=0)
        current_input = current_input.reshape(1, VENTANA_DIAS, N_FEATURES)
        
    return np.array(pronosticos_normalizados), fechas_pronostico 

#DESNORMALIZACIÓN Y SALIDA FINAL (FILTRADO A 2026)

def desnormalizar_y_guardar(pronosticos_norm, scaler, fechas):
    print("Desnormalizando y guardando el pronóstico...")
    
    pronosticos_desnorm = scaler.inverse_transform(pronosticos_norm)
    
    df_pronostico = pd.DataFrame({
        'Fecha': fechas,
        'Temperatura_Predicha': pronosticos_desnorm[:, 0],
        'Lluvia_Predicha': pronosticos_desnorm[:, 1]
    })
    
    df_pronostico['Fecha'] = pd.to_datetime(df_pronostico['Fecha'])
    df_2026 = df_pronostico[df_pronostico['Fecha'].dt.year == 2026].reset_index(drop=True)
    
    nombre_archivo = 'Pronostico_2026_IA.csv'
    df_2026.to_csv(nombre_archivo, index=False)
    
    print(f" ¡Pronóstico de {len(df_2026)} días generado con éxito para el año 2026!")
    print(f" El archivo '{nombre_archivo}' está listo para la Fase 3.")
    print("\nPrimeros 5 días del pronóstico de 2026:")
    print(df_2026.head())
    
#EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    input_seq_inicial, scaler_obj, ultima_fecha_historica = cargar_datos_historicos_y_escalador()
    
    if input_seq_inicial is not None:
        pronosticos_normalizados, fechas_pronostico = predecir_recursivamente(input_seq_inicial, scaler_obj, ultima_fecha_historica)
        
        if pronosticos_normalizados is not None:
            desnormalizar_y_guardar(pronosticos_normalizados, scaler_obj, fechas_pronostico)