"""
Este código es la implementación de la Fase 1: 
Entrenamiento del Modelo LSTM (entrenamiento_modelo.py) del proyecto. 
Su propósito es diseñar, compilar y entrenar la red neuronal para la predicción 
climática de temperatura y precipitación, utilizando el dataset creado previamente.
"""
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# CONFIGURACIÓN DEL MODELO
VENTANA_DIAS = 15
N_FEATURES = 4  #los datos (Temp, Lluvia, Dia_Sin, Dia_Cos)
N_OUTPUTS = 2   # (TARGET_Temp_Manana, TARGET_Lluvia_Manana)

# --- PASO 1: CARGAR Y PREPARAR LOS DATOS
def cargar_y_formatear_datos():
    print("1. Cargando y formateando Dataset_Entrenamiento_IA.csv...")
    try:
        df = pd.read_csv('Dataset_Entrenamiento_IA.csv')
    except FileNotFoundError:
        print(" ERROR: El archivo 'Dataset_Entrenamiento_IA.csv' no se encontró.")
        print("         Asegúrate de ejecutar primero 'preparacion_datos.py'.")
        return None, None, None
        
    # El 80% para entrenamiento, el 20% para validación (prueba)
    split_index = int(len(df) * 0.8)
    
    # Separar entradas (X) y salidas/objetivos (Y)
    
    # X: Columnas de entrada (historia de 15 días)
    X = df.iloc[:, :-N_OUTPUTS].values 
    # Y: Columnas de salida (predicción del día siguiente)
    Y = df.iloc[:, -N_OUTPUTS:].values 

    # IMPORTANTE: Reformatear X a la forma que LSTM espera: [Ejemplos, Pasos de Tiempo, Características]
    X = X.reshape(X.shape[0], VENTANA_DIAS, N_FEATURES)
    print(f"   -> Forma de los datos de entrada (X): {X.shape}")
    print(f"   -> Forma de los datos de salida (Y): {Y.shape}")

    # Separación final
    X_train, X_val = X[:split_index], X[split_index:]
    Y_train, Y_val = Y[:split_index], Y[split_index:]
    
    return X_train, Y_train, X_val, Y_val

#DISEÑO Y COMPILACION EL MODELO LSTM 
def crear_modelo(input_shape):
    print("2. Diseñando la Red Neuronal LSTM...")
    
    model = Sequential([
        # Capa LSTM: Toma la secuencia de 15 días
        LSTM(units=64, 
             activation='tanh', 
             input_shape=input_shape, 
             return_sequences=False), # return_sequences=False porque solo queremos una salida final
        
        # Capa de Regularización para evitar el sobreajuste (overfitting)
        Dropout(0.2), 
        
        # Capa Densa (Feed-Forward)
        Dense(units=32, activation='relu'),
        
        # Capa de Salida: 2 Neuronas para Temp y Lluvia
        Dense(units=N_OUTPUTS, activation='linear')
    ])
    
    # Compilación: Configuración del entrenamiento
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=['mae']) 
    
    print("   -> Modelo compilado. Resumen:")
    model.summary()
    return model

# ENTRENAMIENTO DEL MODELO
def entrenar_modelo(model, X_train, Y_train, X_val, Y_val):
    print("\n3.Iniciando Entrenamiento (Esto tomará unos minutos)...")
    
    #EarlyStopping: Detiene el entrenamiento si el error de validación no mejora.
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    #ModelCheckpoint: Guarda el mejor modelo que se haya visto.
    model_checkpoint = ModelCheckpoint('mejor_modelo_clima.h5', 
                                       monitor='val_loss', 
                                       save_best_only=True, 
                                       mode='min',
                                       verbose=1)

    historial = model.fit(
        X_train, Y_train,
        epochs=50,                  
        batch_size=32,
        validation_data=(X_val, Y_val),
        callbacks=[early_stop, model_checkpoint],
        verbose=1
    )
    
    return historial

# --- PASO 4: EVALUAR Y VISUALIZAR ---
def evaluar_y_graficar(historial):
    print("\n Evaluando el rendimiento...")
    
    # Visualizar la reducción de la pérdida (Loss)
    plt.figure(figsize=(12, 6))
    plt.plot(historial.history['loss'], label='Pérdida (MSE) en Entrenamiento')
    plt.plot(historial.history['val_loss'], label='Pérdida (MSE) en Validación')
    plt.title('Curva de Aprendizaje del Modelo LSTM')
    plt.ylabel('Error Cuadrático Medio (MSE)')
    plt.xlabel('Época')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\n Entrenamiento finalizado.")
    print("   El mejor modelo fue guardado como 'mejor_modelo_clima.h5'.")


#EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val = cargar_y_formatear_datos()
    
    if X_train is not None:
        # La forma de entrada es (15 días, 4 características)
        input_shape = (X_train.shape[1], X_train.shape[2]) 
        
        modelo = crear_modelo(input_shape)
        
        historial_entrenamiento = entrenar_modelo(modelo, X_train, Y_train, X_val, Y_val)
        
        evaluar_y_graficar(historial_entrenamiento)