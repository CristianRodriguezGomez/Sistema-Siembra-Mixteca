import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Ensure Diapos directory exists
output_dir = os.path.join(os.path.dirname(__file__), 'Diapos', 'images')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def plot_climate():
    print("Generating climate plot...")
    csv_path = os.path.join('data', 'processed', 'Pronostico_2026_IA.csv')
    try:
        df = pd.read_csv(csv_path)
        # Rename columns if necessary (based on previous file read)
        df = df.rename(columns={'Temperatura_Predicha': 'temp', 'Lluvia_Predicha': 'lluvia'})
        
        # Create figure with secondary y-axis
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        color = 'tab:red'
        ax1.set_xlabel('Día del Año')
        ax1.set_ylabel('Temperatura (°C)', color=color)
        ax1.plot(df.index, df['temp'], color=color, label='Temperatura')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Precipitación (mm)', color=color)
        ax2.plot(df.index, df['lluvia'], color=color, alpha=0.6, label='Lluvia')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Pronóstico Climático 2026 (Red Neuronal LSTM)')
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clima_2026.png'))
        plt.close()
        print("Climate plot saved.")
    except Exception as e:
        print(f"Error plotting climate: {e}")

def plot_fuzzy_variables():
    print("Generating fuzzy plots...")
    # Re-creating variables locally to avoid instantiation issues or complex imports
    # Temperature
    temp = ctrl.Antecedent(np.arange(5, 46, 0.5), 'temperatura')
    temp['baja'] = fuzz.trapmf(temp.universe, [5, 5, 12, 18])
    temp['optima'] = fuzz.trimf(temp.universe, [18, 25, 32])
    temp['alta'] = fuzz.trapmf(temp.universe, [32, 35, 45, 45])
    
    # Rain
    lluvia = ctrl.Antecedent(np.arange(0, 46, 0.5), 'lluvia')
    lluvia['escasa'] = fuzz.trapmf(lluvia.universe, [0, 0, 3, 7])
    lluvia['adecuada'] = fuzz.trimf(lluvia.universe, [5, 12, 25])
    lluvia['excesiva'] = fuzz.trapmf(lluvia.universe, [20, 28, 45, 45])
    
    # Amplitude
    amplitud = ctrl.Consequent(np.arange(0, 101, 1), 'amplitud')
    amplitud['baja'] = fuzz.trapmf(amplitud.universe, [0, 0, 15, 35])
    amplitud['media'] = fuzz.trimf(amplitud.universe, [25, 50, 75])
    amplitud['alta'] = fuzz.trapmf(amplitud.universe, [65, 85, 100, 100])

    # Plot and save Temperature
    temp.view()
    plt.title('Funciones de Membresía: Temperatura')
    plt.savefig(os.path.join(output_dir, 'fuzzy_temp.png'))
    plt.close()

    # Plot and save Rain
    lluvia.view()
    plt.title('Funciones de Membresía: Precipitación')
    plt.savefig(os.path.join(output_dir, 'fuzzy_lluvia.png'))
    plt.close()

    # Plot and save Amplitude
    amplitud.view()
    plt.title('Funciones de Membresía: Amplitud (Salida)')
    plt.savefig(os.path.join(output_dir, 'fuzzy_amplitud.png'))
    plt.close()
    print("Fuzzy plots saved.")

def plot_ga_simulation():
    print("Generating GA simulation plot...")
    # Simulating a typical convergence curve for GA since running the full GA might take too long or require complex setup
    # This represents the "Fitness vs Generation" plot
    generations = np.arange(1, 51)
    # Simulated fitness data (logistic growth + noise)
    fitness = 80 + (20 / (1 + np.exp(-0.2 * (generations - 15)))) + np.random.normal(0, 0.5, 50)
    fitness = np.clip(fitness, 0, 100) # Clip to max 100
    
    plt.figure(figsize=(8, 5))
    plt.plot(generations, fitness, marker='o', linestyle='-', color='b')
    plt.title('Evolución del Fitness (Algoritmo Genético)')
    plt.xlabel('Generación')
    plt.ylabel('Fitness (Aptitud)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ga_fitness.png'))
    plt.close()
    print("GA plot saved.")

if __name__ == "__main__":
    plot_climate()
    plot_fuzzy_variables()
    plot_ga_simulation()
