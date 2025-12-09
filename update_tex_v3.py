
import os

latex_content = r"""\documentclass{beamer}

\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\graphicspath{{images/}}
\usepackage{ragged2e}

% ====== COLORES CAFES / OSCUROS ======
\definecolor{mybg}{RGB}{24,20,18}
\definecolor{myaccent}{RGB}{214,163,92}
\definecolor{mysecondary}{RGB}{180,120,70}

\usetheme{Madrid}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]

\setbeamercolor{background canvas}{bg=mybg}
\setbeamercolor{normal text}{fg=white}
\setbeamercolor{frametitle}{fg=white,bg=black!70}
\setbeamercolor{title}{fg=white}
\setbeamercolor{structure}{fg=myaccent}
\setbeamercolor{itemize item}{fg=myaccent}
\setbeamercolor{itemize subitem}{fg=mysecondary}

\setkeys{Gin}{width=\linewidth, keepaspectratio}

\title{Sistema de Optimizacion de Siembra \\ para la Mixteca Oaxaquena}
\subtitle{Redes Neuronales, Logica Difusa y Algoritmos Geneticos}
\author{\textbf{Integrantes:}\\
Aneli Arce Jimenez\\
Cristian Rodriguez Gomez\\
Ossiel Alejandro Acevedo Herrera\\
Ramon Aragon Toledo}
\institute{Inteligencia Artificial}
\date{\today}

\begin{document}

%------------------ DIAPOSITIVA 1: PORTADA ------------------
\begin{frame}
  \titlepage
\end{frame}

%------------------ DIAPOSITIVA 2: INTRODUCCION ------------------
\begin{frame}{Introduccion}
  \justifying
  \begin{itemize}
    \item La \textbf{Mixteca Oaxaquena} enfrenta desafios climaticos significativos para la agricultura tradicional de maiz.
    \item Determinar la \textbf{fecha optima de siembra} es crucial para:
    \begin{itemize}
      \item Maximizar el rendimiento del cultivo.
      \item Minimizar el riesgo climatico.
      \item Aprovechar las condiciones de temperatura y lluvia.
    \end{itemize}
    \item Este sistema combina \textbf{tres tecnicas de IA} para encontrar automaticamente la mejor fecha de siembra.
  \end{itemize}
\end{frame}

%------------------ DIAPOSITIVA 3: OBJETIVO ------------------
\begin{frame}{Objetivo del Proyecto}
  \begin{itemize}
    \item \textbf{Desarrollar un sistema inteligente} que determine la ventana de siembra optima para maiz en la Mixteca.
    \item Integrar tres tecnicas complementarias:
    \begin{itemize}
      \item \textbf{Redes Neuronales LSTM}: Prediccion climatica.
      \item \textbf{Logica Difusa}: Evaluacion de condiciones de siembra.
      \item \textbf{Algoritmos Geneticos}: Optimizacion de la fecha.
    \end{itemize}
    \item Proveer al agricultor una \textbf{recomendacion clara y fundamentada} basada en datos.
  \end{itemize}
\end{frame}

%------------------ DIAPOSITIVA 4: ARQUITECTURA GENERAL ------------------
\begin{frame}{Arquitectura del Sistema}
  \centering
  \textbf{Pipeline de 4 Fases}
  
  \vspace{0.5em}
  \begin{columns}[T]
    \begin{column}{0.24\textwidth}
      \centering
      \textcolor{myaccent}{\textbf{Fase 0}}\\
      \small Recoleccion\\de Datos
      \vspace{0.3em}
      
      \footnotesize
      Datos historicos\\
      CONAGUA/SMN\\
      Filtrado y\\
      limpieza
    \end{column}
    \begin{column}{0.24\textwidth}
      \centering
      \textcolor{myaccent}{\textbf{Fase 1}}\\
      \small Red Neuronal\\LSTM
      \vspace{0.3em}
      
      \footnotesize
      Prediccion de\\
      temperatura y\\
      precipitacion\\
      2026
    \end{column}
    \begin{column}{0.24\textwidth}
      \centering
      \textcolor{myaccent}{\textbf{Fase 2}}\\
      \small Sistema\\Difuso
      \vspace{0.3em}
      
      \footnotesize
      Evaluacion de\\
      aptitud para\\
      siembra\\
      (0-100)
    \end{column}
    \begin{column}{0.24\textwidth}
      \centering
      \textcolor{myaccent}{\textbf{Fase 3}}\\
      \small Algoritmo\\Genetico
      \vspace{0.3em}
      
      \footnotesize
      Busqueda del\\
      dia optimo\\
      de siembra
    \end{column}
  \end{columns}
  
  \vspace{1em}
  \footnotesize
  \textcolor{mysecondary}{Fase 0 $\rightarrow$ Fase 1 $\rightarrow$ Fase 2 $\rightarrow$ Fase 3 $\rightarrow$ \textbf{Fecha Optima}}
\end{frame}

%------------------ DIAPOSITIVA 5: ESTRUCTURA DE ARCHIVOS ------------------
\begin{frame}{Estructura del Proyecto}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \textbf{Organizacion Modular:}
      \begin{itemize}
        \item \texttt{main.py}: Punto de entrada
        \item \texttt{data/processed/}: Pronosticos CSV
        \item \texttt{src/neural/}: Red LSTM
        \item \texttt{src/fuzzy/}: Sistema difuso
        \item \texttt{src/optimization/}: Alg. genetico
      \end{itemize}
    \end{column}
    \begin{column}{0.48\textwidth}
      \textbf{Flujo de Datos:}
      \begin{enumerate}
        \item CSV con 365 dias de pronostico
        \item Gestor climatico lee datos
        \item Sistema difuso evalua aptitud
        \item Algoritmo genetico optimiza
        \item Resultado: dia optimo del ano
      \end{enumerate}
    \end{column}
  \end{columns}
\end{frame}

%------------------ DIAPOSITIVA NUEVA: FASE 0 - DATOS ------------------
\begin{frame}{Fase 0: Preparacion de Datos y Entrenamiento}
  \justifying
  \textbf{Origen de los Datos:}
  \begin{itemize}
    \item Datos meteorologicos obtenidos de la estacion de \textbf{Huajuapan de Leon}.
    \item \textbf{Periodo Historico}: 19 anos de registros (hasta 2024).
    \item \textit{Nota: No se incluyen datos de 2025 por no estar disponibles aun.}
  \end{itemize}
  
  \vspace{0.5em}
  \textbf{Procesamiento y Seleccion del Modelo:}
  \begin{enumerate}
    \item \textbf{Limpieza}: Filtrado de ruido y tratamiento de valores nulos en el dataset.
    \item \textbf{Seleccion}: Se evaluaron multiples modelos y se selecciono el que presento el \textbf{mejor comportamiento} (menor error).
    \item \textbf{Entrenamiento}: El modelo ganador se entreno con los datos depurados para generar el archivo de pronosticos 2026.
  \end{enumerate}
\end{frame}

%------------------ DIAPOSITIVA 6: FASE 1 - RED NEURONAL CONCEPTO ------------------
\begin{frame}{Fase 1: Red Neuronal LSTM - Concepto}
  \justifying
  \textbf{Que es una Red LSTM?}
  \begin{itemize}
    \item \textbf{Long Short-Term Memory}: Tipo de red neuronal recurrente.
    \item Especializada en aprender \textbf{patrones temporales} en secuencias de datos.
    \item Capaz de ``recordar'' dependencias a largo plazo.
  \end{itemize}
  
  \vspace{0.5em}
  \textbf{Aplicacion en el Proyecto:}
  \begin{itemize}
    \item \textbf{Entrada}: Datos climaticos historicos (temperatura, lluvia).
    \item \textbf{Salida}: Prediccion de clima para cada dia de 2026.
    \item Genera archivo \texttt{Pronostico\_2026\_IA.csv} con 365 registros.
  \end{itemize}
\end{frame}

%------------------ DIAPOSITIVA 7: FASE 1 - GRAFICO ------------------
\begin{frame}{Fase 1: Pronostico Climatico 2026}
  \centering
  \includegraphics[width=0.95\textwidth]{clima_2026.png}
  
  \vspace{0.5em}
  \footnotesize
  \textcolor{mysecondary}{Prediccion de temperatura y precipitacion para todo el ano.}
\end{frame}

% =================================================================================
% SECCION CODIGO 1
% =================================================================================

% INTRO CODIGO 1
\begin{frame}{Introduccion al Codigo 1: Gestor Climatico}
  \justifying
  \textbf{Proposito:}
  \begin{itemize}
    \item Este modulo actua como el \textbf{puente} entre los datos crudos y el sistema inteligente.
    \item Se encarga de leer el archivo CSV generado por la red neuronal.
    \item Prepara y formatea los datos para que puedan ser consumidos por el sistema difuso.
  \end{itemize}
  
  \vspace{1em}
  \textbf{Ubicacion:} \texttt{src/neural/gestor\_climatico.py}
\end{frame}

% CODIGO 1
\begin{frame}[fragile]{Codigo 1: Gestor Climatico}
  \scriptsize
  \begin{verbatim}
FUNCION obtener_clima_real(dia_inicio, duracion=120):
    // 1. Calcular indices de fechas
    indice_inicio = dia_inicio - 1
    indice_fin = indice_inicio + duracion
    
    // 2. Leer archivo CSV generado por la LSTM
    datos_csv = LEER_CSV("Pronostico_2026_IA.csv")
    
    // 3. Extraer ventana de tiempo
    ventana_datos = datos_csv[indice_inicio : indice_fin]
    
    // 4. Formatear salida
    lista_clima = []
    PARA CADA fila EN ventana_datos:
        lista_clima.AGREGAR({
            'temp': fila['temperatura'],
            'lluvia': fila['precipitacion']
        })
        
    RETORNAR lista_clima
  \end{verbatim}
\end{frame}

%------------------ DIAPOSITIVA 8: FASE 2 - LOGICA DIFUSA CONCEPTO ------------------
\begin{frame}{Fase 2: Sistema de Logica Difusa - Concepto}
  \justifying
  \textbf{Que es la Logica Difusa?}
  \begin{itemize}
    \item Extension de la logica booleana que maneja \textbf{grados de verdad}.
    \item Permite modelar conceptos imprecisos: ``temperatura \textit{optima}'', ``lluvia \textit{adecuada}''.
    \item Ideal para sistemas donde no hay limites exactos.
  \end{itemize}
  
  \vspace{0.5em}
  \textbf{Componentes del Sistema:}
  \begin{itemize}
    \item \textbf{Variables de Entrada}: Temperatura (C), Precipitacion (mm).
    \item \textbf{Variable de Salida}: Amplitud de siembra (0-100).
    \item \textbf{Funciones de Membresia}: Trapezoidal y triangular.
    \item \textbf{Reglas de Inferencia}: 9 reglas IF-THEN.
  \end{itemize}
\end{frame}

%------------------ DIAPOSITIVA 9: FASE 2 - GRAFICOS ENTRADA ------------------
\begin{frame}{Fase 2: Variables de Entrada (Funciones de Membresia)}
  \centering
  \begin{columns}
    \begin{column}{0.5\textwidth}
      \centering
      \textbf{Temperatura}\\
      \includegraphics[width=\linewidth]{fuzzy_input_temperatura.png}
    \end{column}
    \begin{column}{0.5\textwidth}
      \centering
      \textbf{Precipitacion}\\
      \includegraphics[width=\linewidth]{fuzzy_input_lluvia.png}
    \end{column}
  \end{columns}
  
  \vspace{1em}
  \justifying
  \footnotesize
  \textcolor{mysecondary}{Definen los rangos para categorias como "Baja", "Optima", "Alta" (Temp) y "Escasa", "Adecuada", "Excesiva" (Lluvia).}
\end{frame}

%------------------ DIAPOSITIVA 10: FASE 2 - GRAFICO SALIDA ------------------
\begin{frame}{Fase 2: Variable de Salida (Amplitud)}
  \centering
  \includegraphics[width=0.7\textwidth]{fuzzy_output_amplitud.png}
  
  \vspace{1em}
  \justifying
  \textbf{Interpretacion del Score (0-100):}
  \begin{itemize}
    \item \textbf{0-35 (Baja)}: Condiciones adversas, no sembrar.
    \item \textbf{25-75 (Media)}: Condiciones aceptables con riesgo.
    \item \textbf{65-100 (Alta)}: Condiciones ideales para siembra.
  \end{itemize}
\end{frame}

% =================================================================================
% SECCION CODIGO 2
% =================================================================================

% INTRO CODIGO 2
\begin{frame}{Introduccion al Codigo 2: Sistema Difuso}
  \justifying
  \textbf{Proposito:}
  \begin{itemize}
    \item Aqui se define el \textbf{"cerebro"} de evaluacion del sistema.
    \item Se configuran las variables linguisticas (temperatura, lluvia) y sus rangos.
    \item Se establecen las \textbf{reglas de inferencia} que determinan que tan bueno es un dia para sembrar basandose en el conocimiento experto.
  \end{itemize}
  
  \vspace{1em}
  \textbf{Ubicacion:} \texttt{src/fuzzy/fuzzy\_system.py}
\end{frame}

% CODIGO 2
\begin{frame}[fragile]{Codigo 2: Sistema Difuso}
  \scriptsize
  \begin{verbatim}
// 1. Definir Antecedentes (Entradas) y Consecuente (Salida)
temperatura = Antecedent(rango=[5, 45], etiqueta='temp')
lluvia = Antecedent(rango=[0, 45], etiqueta='lluvia')
amplitud = Consequent(rango=[0, 100], etiqueta='amplitud')

// 2. Definir Funciones de Membresia (Ejemplo Temperatura)
temperatura['baja'] = trapezoidal([5, 5, 12, 18])
temperatura['optima'] = triangular([18, 25, 32])
temperatura['alta'] = trapezoidal([32, 35, 45, 45])

// 3. Definir Reglas (Base de Conocimiento)
regla_ideal = Rule(lluvia['adecuada'] & temp['optima'], amplitud['alta'])
regla_mala = Rule(lluvia['escasa'] & temp['alta'], amplitud['baja'])
// ... (Total 9 reglas)

// 4. Crear Sistema de Control
sistema = ControlSystem([regla_ideal, regla_mala, ...])
simulacion = ControlSystemSimulation(sistema)
  \end{verbatim}
\end{frame}

% =================================================================================
% SECCION CODIGO 3
% =================================================================================

% INTRO CODIGO 3
\begin{frame}{Introduccion al Codigo 3: Test Visual Difuso}
  \justifying
  \textbf{Proposito:}
  \begin{itemize}
    \item Un script de utilidad para \textbf{verificar visualmente} que el sistema difuso funciona correctamente.
    \item Permite inyectar valores manuales de temperatura y lluvia.
    \item Ayuda a depurar y entender que reglas se estan activando en casos especificos.
  \end{itemize}
  
  \vspace{1em}
  \textbf{Ubicacion:} \texttt{test\_fuzzy\_visual.py}
\end{frame}

% CODIGO 3
\begin{frame}[fragile]{Codigo 3: Test Visual Difuso}
  \scriptsize
  \begin{verbatim}
IMPORTAR sistema_global DESDE src.fuzzy.fuzzy_system

// Valores de prueba manuales
lluvia_test = 20
temp_test = 25

IMPRIMIR "Diagnostico de Logica Difusa"

// 1. Inyectar valores al sistema
sistema_global.input['lluvia'] = lluvia_test
sistema_global.input['temperatura'] = temp_test

// 2. Calcular (Fuzzificacion -> Inferencia -> Defuzzificacion)
sistema_global.compute()

// 3. Obtener resultado
resultado = sistema_global.output['amplitud']
IMPRIMIR "Score obtenido: " + resultado

// 4. Visualizar reglas activadas
PARA CADA regla EN sistema_global.rules:
    IMPRIMIR regla
  \end{verbatim}
\end{frame}

%------------------ DIAPOSITIVA 11: FASE 3 - ALGORITMO GENETICO ------------------
\begin{frame}{Fase 3: Algoritmo Genetico - Concepto}
  \justifying
  \textbf{Que es un Algoritmo Genetico?}
  \begin{itemize}
    \item Tecnica de \textbf{optimizacion inspirada en la evolucion natural}.
    \item Poblacion de soluciones que ``evolucionan'' hacia el optimo.
    \item Operadores: seleccion, cruce y mutacion.
  \end{itemize}
  
  \vspace{0.5em}
  \textbf{Aplicacion en el Proyecto:}
  \begin{itemize}
    \item \textbf{Cromosoma}: Un dia del ano (1-365).
    \item \textbf{Fitness}: Suma de aptitudes del ciclo de cultivo (120 dias).
    \item \textbf{Objetivo}: Encontrar el dia con maximo fitness acumulado.
  </end{itemize}
\end{frame}

%------------------ DIAPOSITIVA 12: FASE 3 - GRAFICO FITNESS ------------------
\begin{frame}{Fase 3: Evolucion del Fitness}
  \centering
  \includegraphics[width=0.8\textwidth]{ga_fitness.png}
  
  \vspace{0.5em}
  \footnotesize
  \textcolor{mysecondary}{Mejora progresiva de la solucion a traves de las generaciones.}
\end{frame}

% =================================================================================
% SECCION CODIGO 4
% =================================================================================

% INTRO CODIGO 4
\begin{frame}{Introduccion al Codigo 4: Algoritmo Genetico}
  \justifying
  \textbf{Proposito:}
  \begin{itemize}
    \item El \textbf{motor de optimizacion} del proyecto.
    \item En lugar de probar cada dia del ano uno por uno, evoluciona una poblacion de fechas.
    \item Utiliza una funcion de fitness que evalua el clima de los 120 dias siguientes a la fecha propuesta.
  \end{itemize}
  
  \vspace{1em}
  \textbf{Ubicacion:} \texttt{src/optimization/algoritmo\_genetico.py}
\end{frame}

% CODIGO 4
\begin{frame}[fragile]{Codigo 4: Algoritmo Genetico}
  \scriptsize
  \begin{verbatim}
FUNCION fitness_func(solucion, indice):
    dia_siembra = solucion[0]
    
    // Penalizar si esta fuera de rango (ej. fin de ano)
    SI dia_siembra > 240: RETORNAR -9999
    
    // Obtener clima para los siguientes 120 dias
    datos = obtener_clima_real(dia_siembra, 120)
    
    score_total = 0
    PARA CADA dia EN datos:
        // Evaluar cada dia con logica difusa
        aptitud = calcular_aptitud(dia.lluvia, dia.temp)
        score_total += aptitud
        
    RETORNAR score_total

FUNCION correr_optimizacion():
    ga = GA(num_generaciones=50, 
            poblacion=20, 
            fitness_func=fitness_func,
            tipo_gen=ENTERO)
    ga.run()
    RETORNAR ga.best_solution()
  \end{verbatim}
\end{frame}

% =================================================================================
% SECCION CODIGO 5
% =================================================================================

% INTRO CODIGO 5
\begin{frame}{Introduccion al Codigo 5: Graficar Panorama}
  \justifying
  \textbf{Proposito:}
  \begin{itemize}
    \item Herramienta de \textbf{validacion y analisis global}.
    \item Calcula la aptitud de siembra para \textbf{todos} los dias del ano (fuerza bruta).
    \item Genera una grafica que permite ver los picos de aptitud y verificar si el algoritmo genetico encontro el verdadero optimo.
  \end{itemize}
  
  \vspace{1em}
  \textbf{Ubicacion:} \texttt{graficar\_panorama.py}
\end{frame}

% CODIGO 5
\begin{frame}[fragile]{Codigo 5: Graficar Panorama (Validacion)}
  \scriptsize
  \begin{verbatim}
// Objetivo: Comparar GA vs Fuerza Bruta (todos los dias)

listas_scores = []
dias_del_ano = RANGO(1, 365)

IMPRIMIR "Generando panorama completo..."

PARA dia EN dias_del_ano:
    // Calcular score real para cada dia posible
    score = obtener_score_del_dia(dia)
    listas_scores.AGREGAR(score)

// Graficar la curva completa
PLOT(dias_del_ano, listas_scores)
TITULO("Aptitud de Siembra para todo el ano")
MOSTRAR_GRAFICA()
  \end{verbatim}
\end{frame}

% =================================================================================
% SECCION CODIGO 6
% =================================================================================

% INTRO CODIGO 6
\begin{frame}{Introduccion al Codigo 6: Mocks (Simulacion)}
  \justifying
  \textbf{Proposito:}
  \begin{itemize}
    \item Modulos de \textbf{prueba y simulacion}.
    \item Generan datos falsos pero realistas para probar el sistema sin depender de archivos externos o conexiones.
    \item Esenciales para el desarrollo agil y pruebas unitarias.
  \end{itemize}
  
  \vspace{1em}
  \textbf{Ubicacion:} \texttt{src/mocks.py}
\end{frame}

% CODIGO 6
\begin{frame}[fragile]{Codigo 6: Mocks (Simulacion)}
  \scriptsize
  \begin{verbatim}
// Utilidad para pruebas sin datos reales o dependencias complejas

FUNCION predecir_clima_simulado(dia_inicio):
    // Genera datos aleatorios pero realistas
    clima = []
    PARA i EN RANGO(120):
        clima.AGREGAR({
            'temp': RANDOM(15, 32),
            'lluvia': RANDOM(0, 50)
        })
    RETORNAR clima

FUNCION evaluar_riesgo_simulado(temp, lluvia):
    // Reglas simples para pruebas unitarias
    riesgo = 0
    SI temp > 30: riesgo += 50
    SI lluvia < 5: riesgo += 50
    RETORNAR riesgo
  \end{verbatim}
\end{frame}

% =================================================================================
% SECCION CODIGO 7
% =================================================================================

% INTRO CODIGO 7
\begin{frame}{Introduccion al Codigo 7: Main}
  \justifying
  \textbf{Proposito:}
  \begin{itemize}
    \item El \textbf{punto de entrada} principal de la aplicacion.
    \item Orquesta todo el flujo: inicializa componentes, llama a la optimizacion y presenta los resultados.
    \item Convierte el indice numerico del dia (ej. 150) a una fecha legible (ej. 30 de Mayo).
  \end{itemize}
  
  \vspace{1em}
  \textbf{Ubicacion:} \texttt{main.py}
\end{frame}

% CODIGO 7
\begin{frame}[fragile]{Codigo 7: Main (Punto de Entrada)}
  \scriptsize
  \begin{verbatim}
IMPORTAR correr_optimizacion
IMPORTAR datetime

FUNCION main():
    IMPRIMIR "--- SISTEMA DE OPTIMIZACION DE SIEMBRA ---"
    
    // 1. Ejecutar el Algoritmo Genetico
    mejor_dia_indice = correr_optimizacion()
    
    // 2. Convertir indice (1-365) a Fecha
    fecha_inicio = FECHA(2026, 1, 1)
    fecha_optima = fecha_inicio + DIAS(mejor_dia_indice - 1)
    
    // 3. Mostrar Resultados
    IMPRIMIR "Mejor dia para sembrar: Dia " + mejor_dia_indice
    IMPRIMIR "Fecha recomendada: " + fecha_optima
    
SI __name__ == "__main__":
    main()
  \end{verbatim}
\end{frame}

%------------------ DIAPOSITIVA 13: RESULTADOS ------------------
\begin{frame}{Resultados del Sistema}
  \justifying
  \textbf{Comportamiento Observado:}
  \begin{itemize}
    \item El algoritmo genetico \textbf{converge consistentemente} hacia fechas en temporada de lluvias.
    \item Las fechas recomendadas coinciden con el conocimiento tradicional campesino.
    \item El fitness muestra incremento progresivo a lo largo de las generaciones.
  \end{itemize}
  
  \vspace{0.5em}
  \textbf{Ventajas del Sistema:}
  \begin{itemize}
    \item \textbf{Automatizado}: No requiere intervencion manual.
    \item \textbf{Fundamentado}: Basado en pronosticos climaticos.
    \item \textbf{Adaptable}: Puede actualizarse con nuevos datos.
    \item \textbf{Interpretable}: La logica difusa explica las decisiones.
  \end{itemize}
\end{frame}

%------------------ DIAPOSITIVA 14: CONCLUSIONES ------------------
\begin{frame}{Conclusiones}
  \justifying
  \textbf{Logros Tecnicos:}
  \begin{itemize}
    \item Se implemento exitosamente un sistema hibrido de \textbf{3 tecnicas de IA}.
    \item La \textbf{Red LSTM} logra predicciones climaticas para todo el ano.
    \item El \textbf{Sistema Difuso} traduce condiciones climaticas en aptitud de siembra.
    \item El \textbf{Algoritmo Genetico} encuentra eficientemente la fecha optima.
  \end{itemize}
  
  \vspace{0.5em}
  \textbf{Impacto Potencial:}
  \begin{itemize}
    \item Herramienta de apoyo para agricultores de la Mixteca.
    \item Reduccion del riesgo de perdida de cosechas.
    \item Modelo replicable para otras regiones agricolas.
  \end{itemize}
\end{frame}

%------------------ DIAPOSITIVA 15: GRACIAS ------------------
\begin{frame}
  \centering
  \vspace{2em}
  {\Huge \textcolor{myaccent}{Gracias!}}
  
  \vspace{1em}
  {\Large Sistema de Optimizacion de Siembra\\para la Mixteca Oaxaquena}
  
  \vspace{1.5em}
  \textbf{Preguntas?}
\end{frame}

\end{document}
"""

with open("Diapos/presentacion_siembra_mixteca.tex", "w", encoding="utf-8") as f:
    f.write(latex_content)

print("Archivo LaTeX actualizado correctamente.")
