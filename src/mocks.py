import random

# Simula la Red Neuronal (Compañera 1)
def predecir_clima_simulado(dia_inicio, duracion=120):
    clima = []
    for _ in range(duracion):
        clima.append({
            'temp': random.uniform(15, 32), # Temp aleatoria realista
            'lluvia': random.uniform(0, 50) if random.random() > 0.6 else 0
        })
    return clima

# Simula la Lógica Difusa (Compañero 2)
def evaluar_riesgo_simulado(temp, lluvia):
    # Regla simple para probar: Si no llueve y hace calor, es malo.
    riesgo = 0
    if temp > 30: riesgo += 50
    if lluvia < 5: riesgo += 50
    return min(riesgo, 100)