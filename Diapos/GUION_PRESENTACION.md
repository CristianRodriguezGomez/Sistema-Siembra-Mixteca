# GUIÓN DE PRESENTACIÓN
## Sistema de Optimización de Siembra para la Mixteca Oaxaqueña

**Duración estimada:** 25-30 minutos  
**Total de diapositivas:** 34

---

## DISTRIBUCIÓN DE ROLES

| Integrante | Fase Principal | Diapositivas |
|------------|----------------|--------------|
| **Ossiel Alejandro Acevedo Herrera** | Fase 0: Datos | Portada, Introducción, Objetivo, Arquitectura, Estructura, Fase 0, Pipeline de Datos |
| **Aneli Arce Jiménez** | Fase 1: Red LSTM | LSTM Concepto, Pronóstico, Gestor Climático |
| **Ramón Aragón Toledo** | Fase 2: Lógica Difusa | Sistema Difuso Concepto, Variables Entrada/Salida, Código Difuso, Test Visual |
| **Cristian Rodríguez Gómez** | Fase 3: Optimización | Algoritmo Genético, PSO, Códigos de Optimización, Resultados, Conclusiones |

---

# PARTE 1: OSSIEL ALEJANDRO ACEVEDO HERRERA
## (Introducción General + Fase 0: Preparación de Datos)

---

### DIAPOSITIVA 1: PORTADA
**[OSSIEL]**

> "Buenos días/tardes. Somos el equipo conformado por Aneli, Cristian, Ramón y un servidor Ossiel. El día de hoy les presentaremos nuestro proyecto: **Sistema de Optimización de Siembra para la Mixteca Oaxaqueña**, el cual integra tres técnicas de Inteligencia Artificial: Redes Neuronales, Lógica Difusa y Algoritmos Genéticos."

---

### DIAPOSITIVA 2: INTRODUCCIÓN
**[OSSIEL]**

> "Para contextualizar el problema: La Mixteca Oaxaqueña es una región que enfrenta **desafíos climáticos muy significativos** para la agricultura tradicional de maíz.
>
> Determinar la **fecha óptima de siembra** es crucial por tres razones principales:
> - Primero, para **maximizar el rendimiento** del cultivo.
> - Segundo, para **minimizar el riesgo climático** ante sequías o lluvias excesivas.
> - Y tercero, para **aprovechar al máximo** las condiciones de temperatura y precipitación.
>
> Nuestro sistema combina **tres técnicas de IA** para encontrar automáticamente la mejor fecha de siembra, sin depender únicamente del conocimiento tradicional."

---

### DIAPOSITIVA 3: OBJETIVO DEL PROYECTO
**[OSSIEL]**

> "El objetivo principal del proyecto es **desarrollar un sistema inteligente** que determine la ventana de siembra óptima para maíz en la Mixteca.
>
> Para lograrlo, integramos tres técnicas complementarias:
> - **Redes Neuronales LSTM** para la predicción climática.
> - **Lógica Difusa** para evaluar las condiciones de siembra.
> - **Algoritmos Genéticos** para optimizar y encontrar la mejor fecha.
>
> El resultado final es proveer al agricultor una **recomendación clara y fundamentada**, basada completamente en datos."

---

### DIAPOSITIVA 4: ARQUITECTURA DEL SISTEMA
**[OSSIEL]**

> "La arquitectura del sistema se divide en un **pipeline de 4 fases**:
>
> - **Fase 0** que es la que yo desarrollé: Se encarga de la **recolección de datos** históricos de fuentes como CONAGUA y NASA POWER, incluyendo el filtrado y limpieza.
>
> - **Fase 1** que desarrolló Aneli: Utiliza una **Red Neuronal LSTM** para predecir la temperatura y precipitación de todo el año 2026.
>
> - **Fase 2** que desarrolló Ramón: Implementa un **Sistema de Lógica Difusa** que evalúa qué tan apto es cada día para sembrar, con un puntaje de 0 a 100.
>
> - **Fase 3** que desarrolló Cristian: Aplica un **Algoritmo Genético** (y también PSO) para buscar el día óptimo de siembra.
>
> El flujo va de Fase 0 hasta Fase 3, y el resultado final es la **fecha óptima recomendada**."

---

### DIAPOSITIVA 5: ESTRUCTURA DEL PROYECTO
**[OSSIEL]**

> "A nivel técnico, el proyecto tiene una **organización modular**:
>
> - El archivo `main.py` es el punto de entrada.
> - La carpeta `data/processed/` contiene los pronósticos en formato CSV.
> - En `src/neural/` está la implementación de la Red LSTM.
> - En `src/fuzzy/` está el sistema de lógica difusa.
> - Y en `src/optimization/` están los algoritmos genético y PSO.
>
> El flujo de datos es: el CSV con 365 días de pronóstico es leído por el gestor climático, luego el sistema difuso evalúa la aptitud, el algoritmo genético optimiza, y finalmente obtenemos el día óptimo del año."

---

### DIAPOSITIVA 6: FASE 0 - PREPARACIÓN DE DATOS Y ENTRENAMIENTO
**[OSSIEL]**

> "Ahora les explico la **Fase 0**, que fue mi responsabilidad.
>
> **El origen de los datos** proviene de la estación meteorológica de **Huajuapan de León**. Contamos con un **período histórico de 19 años** de registros, hasta 2024. Es importante mencionar que no se incluyen datos de 2025 porque aún no están disponibles completamente.
>
> El **procesamiento** que realicé incluyó tres pasos:
> 1. **Limpieza**: Filtrado de ruido y tratamiento de valores nulos en el dataset.
> 2. **Selección del modelo**: Evalué múltiples arquitecturas de redes neuronales y seleccioné la que presentó el **menor error** en las predicciones.
> 3. **Entrenamiento**: El modelo ganador se entrenó con los datos depurados para generar el archivo de pronósticos 2026."

---

### DIAPOSITIVA 31: PIPELINE COMPLETO DE DATOS
**[OSSIEL]**

> "Para que tengan una visión más clara del flujo de archivos, les muestro el **pipeline completo**:
>
> Del lado izquierdo tenemos los **scripts de procesamiento**:
> 1. `preparacion_datos.py`: Descarga los datos de la API de NASA POWER, los limpia y prepara las variables cíclicas.
> 2. `entrenamiento_modelo.py`: Diseña y entrena la LSTM usando Keras.
> 3. `generar_pronostico.py`: Usa el modelo entrenado para predecir el clima de 2026.
>
> Del lado derecho están los **archivos generados**:
> - `Reporte_Humano_Huajuapan.csv`: Datos históricos limpios para validación visual.
> - `Dataset_Entrenamiento_IA.csv`: Ventanas de 15 días formateadas para la LSTM.
> - `mejor_modelo_clima.h5`: La red neuronal entrenada con todos sus pesos.
> - `Pronostico_2026_IA.csv`: La predicción diaria que consume el algoritmo genético."

---

### DIAPOSITIVA 32: DETALLE - PREPARACIÓN DE DATOS (NASA POWER)
**[OSSIEL]**

> "Profundizando en la **fuente de datos**:
>
> Utilizamos la **API de NASA POWER**, que proporciona datos satelitales de temperatura y precipitación. La ubicación específica es Huajuapan de León, Oaxaca, y el período abarca 19 años de historia.
>
> El **procesamiento aplicado** incluye:
> 1. **Limpieza**: Eliminación de valores nulos y datos anómalos.
> 2. **Codificación Cíclica**: Transformamos las fechas a valores de Seno y Coseno para que la red neuronal pueda capturar la estacionalidad del clima.
> 3. **Normalización**: Escalamos todas las variables para que estén en rangos compatibles con la red neuronal.
> 4. **Ventaneo**: Creamos secuencias de 15 días consecutivos, que son la entrada que la LSTM espera recibir.
>
> Con esto, le paso la palabra a Aneli para que nos explique la Fase 1."

---

# PARTE 2: ANELI ARCE JIMÉNEZ
## (Fase 1: Red Neuronal LSTM)

---

### DIAPOSITIVA 7: FASE 1 - RED NEURONAL LSTM - CONCEPTO
**[ANELI]**

> "Gracias Ossiel. Ahora les explicaré la **Fase 1**, que corresponde a la red neuronal.
>
> **¿Qué es una Red LSTM?** Las siglas significan **Long Short-Term Memory**, y es un tipo especial de red neuronal recurrente. Está especializada en aprender **patrones temporales** en secuencias de datos, y lo más importante: es capaz de 'recordar' dependencias a largo plazo.
>
> En nuestro proyecto, la LSTM funciona así:
> - **Entrada**: Recibe los datos climáticos históricos de temperatura y lluvia.
> - **Salida**: Genera una predicción del clima para cada día de 2026.
> - El resultado se guarda en el archivo `Pronostico_2026_IA.csv` con 365 registros, uno por cada día del año."

---

### DIAPOSITIVA 8: FASE 1 - PRONÓSTICO CLIMÁTICO 2026
**[ANELI]**

> "Aquí pueden observar el **resultado visual** de la predicción de la red LSTM.
>
> La gráfica muestra la **predicción de temperatura y precipitación** para todo el año 2026. Pueden notar cómo el modelo captura los patrones estacionales: temperaturas más altas en primavera-verano y el período de lluvias concentrado en los meses de junio a septiembre.
>
> Esta información es crucial porque alimenta directamente al sistema de lógica difusa que explicará Ramón."

---

### DIAPOSITIVA 9: INTRODUCCIÓN AL CÓDIGO 1 - GESTOR CLIMÁTICO
**[ANELI]**

> "Ahora les explico el **Gestor Climático**, que es un módulo clave de la Fase 1.
>
> Su propósito es actuar como el **puente** entre los datos crudos y el sistema inteligente. Se encarga de leer el archivo CSV que generó la red neuronal y preparar los datos en el formato que necesita el sistema difuso.
>
> El archivo se encuentra en `src/neural/gestor_climatico.py`."

---

### DIAPOSITIVA 10: CÓDIGO 1 - GESTOR CLIMÁTICO
**[ANELI]**

> "Aquí vemos el pseudocódigo del gestor climático.
>
> La función `obtener_clima_real` recibe dos parámetros: el día de inicio y la duración del ciclo de cultivo, que por defecto son 120 días.
>
> El proceso es:
> 1. Calcula los índices de las fechas que necesita extraer.
> 2. Lee el archivo CSV generado por la LSTM.
> 3. Extrae la ventana de tiempo correspondiente.
> 4. Formatea la salida como una lista de diccionarios, donde cada día tiene su temperatura y lluvia.
>
> Este módulo es fundamental porque permite que el algoritmo genético consulte el clima de cualquier período del año. Ahora le cedo la palabra a Ramón para la Fase 2."

---

# PARTE 3: RAMÓN ARAGÓN TOLEDO
## (Fase 2: Sistema de Lógica Difusa)

---

### DIAPOSITIVA 11: FASE 2 - SISTEMA DE LÓGICA DIFUSA - CONCEPTO
**[RAMÓN]**

> "Gracias Aneli. Yo les explicaré la **Fase 2**, que corresponde al sistema de Lógica Difusa.
>
> **¿Qué es la Lógica Difusa?** Es una extensión de la lógica booleana tradicional que maneja **grados de verdad**. En lugar de solo 'verdadero' o 'falso', podemos tener valores intermedios. Esto nos permite modelar conceptos imprecisos como 'temperatura óptima' o 'lluvia adecuada', que no tienen límites exactos.
>
> Los **componentes de nuestro sistema** son:
> - **Variables de entrada**: Temperatura en grados Celsius y Precipitación en milímetros.
> - **Variable de salida**: Amplitud de siembra, con un puntaje de 0 a 100.
> - **Funciones de membresía**: Usamos formas trapezoidales y triangulares.
> - **Reglas de inferencia**: Definimos 9 reglas del tipo IF-THEN que capturan el conocimiento experto sobre agricultura."

---

### DIAPOSITIVA 12: FASE 2 - VARIABLES DE ENTRADA
**[RAMÓN]**

> "Aquí pueden ver las **funciones de membresía** de las variables de entrada.
>
> Para la **Temperatura**, definimos tres categorías:
> - 'Baja': cuando está entre 5 y 18 grados.
> - 'Óptima': el pico está alrededor de 25 grados.
> - 'Alta': temperaturas superiores a 32 grados.
>
> Para la **Precipitación**, también tenemos tres categorías:
> - 'Escasa': muy poca lluvia, lo cual es malo para el cultivo.
> - 'Adecuada': el rango ideal de precipitación.
> - 'Excesiva': demasiada lluvia, que puede dañar los cultivos.
>
> Estas funciones permiten que valores numéricos como '28 grados' se traduzcan a términos lingüísticos como '60% óptima y 40% alta'."

---

### DIAPOSITIVA 13: FASE 2 - VARIABLE DE SALIDA (AMPLITUD)
**[RAMÓN]**

> "La variable de **salida** es la Amplitud de Siembra, que va de 0 a 100.
>
> La interpretación del puntaje es:
> - **0 a 35 (Baja)**: Condiciones adversas, no se recomienda sembrar.
> - **25 a 75 (Media)**: Condiciones aceptables pero con cierto riesgo.
> - **65 a 100 (Alta)**: Condiciones ideales para la siembra.
>
> Este puntaje es lo que el algoritmo genético va a acumular para cada día, sumando la aptitud de los 120 días del ciclo de cultivo."

---

### DIAPOSITIVA 14: INTRODUCCIÓN AL CÓDIGO 2 - SISTEMA DIFUSO
**[RAMÓN]**

> "Ahora veamos el código del sistema difuso.
>
> Este módulo define el **'cerebro' de evaluación** del sistema. Aquí configuramos las variables lingüísticas con sus rangos, y establecemos las **reglas de inferencia** que determinan qué tan bueno es un día para sembrar.
>
> El archivo está en `src/fuzzy/fuzzy_system.py`."

---

### DIAPOSITIVA 15: CÓDIGO 2 - SISTEMA DIFUSO
**[RAMÓN]**

> "El pseudocódigo muestra los cuatro pasos principales:
>
> 1. **Definir Antecedentes y Consecuente**: Creamos las variables de temperatura, lluvia y amplitud con sus rangos.
>
> 2. **Definir Funciones de Membresía**: Por ejemplo, para temperatura definimos 'baja' como trapezoidal de 5 a 18 grados, 'óptima' como triangular centrada en 25, y 'alta' de 32 en adelante.
>
> 3. **Definir las Reglas**: La regla ideal dice: SI la lluvia es adecuada Y la temperatura es óptima, ENTONCES la amplitud es alta. La regla mala dice: SI la lluvia es escasa Y la temperatura es alta, ENTONCES la amplitud es baja. En total tenemos 9 reglas que cubren todas las combinaciones.
>
> 4. **Crear el Sistema de Control**: Combinamos todas las reglas y creamos una simulación que podemos ejecutar."

---

### DIAPOSITIVA 16: INTRODUCCIÓN AL CÓDIGO 3 - TEST VISUAL
**[RAMÓN]**

> "También desarrollé un script de **prueba visual** llamado `test_fuzzy_visual.py`.
>
> Su propósito es **verificar visualmente** que el sistema difuso funciona correctamente. Permite inyectar valores manuales de temperatura y lluvia para ver qué puntaje genera el sistema, y ayuda a entender qué reglas se están activando en casos específicos."

---

### DIAPOSITIVA 17: CÓDIGO 3 - TEST VISUAL DIFUSO
**[RAMÓN]**

> "El código del test es sencillo:
>
> Primero importamos el sistema difuso global. Luego definimos valores de prueba, por ejemplo, lluvia de 20 mm y temperatura de 25 grados.
>
> Inyectamos estos valores al sistema, ejecutamos el cálculo que hace la fuzzificación, inferencia y defuzzificación, y obtenemos el resultado.
>
> Finalmente, podemos visualizar qué reglas se activaron. Esto fue muy útil durante el desarrollo para verificar que las reglas estuvieran correctamente configuradas.
>
> Ahora le paso la palabra a Cristian para la Fase 3."

---

# PARTE 4: CRISTIAN RODRÍGUEZ GÓMEZ
## (Fase 3: Algoritmos de Optimización + Cierre)

---

### DIAPOSITIVA 18: FASE 3 - ALGORITMO GENÉTICO - CONCEPTO
**[CRISTIAN]**

> "Gracias Ramón. Finalmente, les explicaré la **Fase 3**, que es donde todo el sistema se integra para encontrar la fecha óptima.
>
> **¿Qué es un Algoritmo Genético?** Es una técnica de optimización **inspirada en la evolución natural**. Trabajamos con una población de soluciones que 'evolucionan' hacia el óptimo mediante operadores de selección, cruce y mutación.
>
> En nuestro proyecto:
> - El **cromosoma** es simplemente un día del año, del 1 al 365.
> - El **fitness** o aptitud se calcula sumando los puntajes del sistema difuso para los 120 días del ciclo de cultivo.
> - El **objetivo** es encontrar el día que tenga el máximo fitness acumulado."

---

### DIAPOSITIVA 19: FASE 3 - EVOLUCIÓN DEL FITNESS
**[CRISTIAN]**

> "Esta gráfica muestra la **evolución del fitness** a lo largo de las generaciones.
>
> Pueden observar cómo la solución mejora progresivamente. En las primeras generaciones hay mucha variabilidad porque el algoritmo está explorando el espacio de búsqueda. Conforme avanzan las generaciones, la curva se estabiliza porque ya encontró una buena región.
>
> Esta convergencia nos indica que el algoritmo está funcionando correctamente."

---

### DIAPOSITIVA 20: INTRODUCCIÓN AL CÓDIGO 4 - ALGORITMO GENÉTICO
**[CRISTIAN]**

> "El algoritmo genético es el **motor de optimización** del proyecto.
>
> En lugar de probar cada día del año uno por uno, lo cual sería fuerza bruta, evolucionamos una población de fechas candidatas. Cada candidato se evalúa usando la función de fitness que consulta el clima de los 120 días siguientes.
>
> El archivo está en `src/optimization/algoritmo_genetico.py`."

---

### DIAPOSITIVA 21: CÓDIGO 4 - ALGORITMO GENÉTICO
**[CRISTIAN]**

> "Veamos el pseudocódigo:
>
> La función `fitness_func` recibe una solución (un día de siembra). Primero verifica restricciones: si el día es mayor a 240, lo penaliza porque no habría tiempo de cosechar antes de fin de año.
>
> Luego obtiene el clima para los siguientes 120 días usando el gestor climático de Aneli. Para cada día, calcula la aptitud con el sistema difuso de Ramón y suma todo.
>
> La función `correr_optimizacion` configura el algoritmo genético con 50 generaciones, 20 individuos por población, y lo ejecuta. Al final retorna la mejor solución encontrada."

---

### DIAPOSITIVA 22: INTRODUCCIÓN AL CÓDIGO 5 - GRAFICAR PANORAMA
**[CRISTIAN]**

> "También desarrollé una herramienta de **validación** llamada `graficar_panorama.py`.
>
> Su propósito es calcular la aptitud de siembra para **todos** los días del año usando fuerza bruta. Esto genera una curva completa que nos permite verificar si el algoritmo genético realmente encontró el óptimo global, o si se quedó en un óptimo local."

---

### DIAPOSITIVA 23: CÓDIGO 5 - GRAFICAR PANORAMA
**[CRISTIAN]**

> "El código es directo: iteramos por cada día del año del 1 al 365, calculamos el score para ese día, y guardamos todos los valores. Después graficamos la curva completa.
>
> Esta validación es importante porque nos da confianza en que el algoritmo genético no nos está engañando."

---

### DIAPOSITIVA 24-27: CÓDIGOS MOCKS Y MAIN
**[CRISTIAN]**

> "Brevemente menciono dos módulos adicionales:
>
> Los **Mocks** en `src/mocks.py` son módulos de prueba que generan datos simulados. Son esenciales para desarrollo y pruebas unitarias cuando no tenemos acceso a los datos reales.
>
> El **Main** en `main.py` es el punto de entrada de la aplicación. Orquesta todo el flujo: ejecuta la optimización, convierte el índice numérico del día a una fecha legible, y presenta los resultados al usuario."

---

### DIAPOSITIVA 28: RESULTADOS DEL SISTEMA
**[CRISTIAN]**

> "Pasemos a los **resultados observados**:
>
> El algoritmo genético **converge consistentemente** hacia fechas en la temporada de lluvias, generalmente entre mayo y junio. Esto coincide con el conocimiento tradicional campesino de la región.
>
> Las **ventajas del sistema** son:
> - Es **automatizado**: no requiere intervención manual.
> - Está **fundamentado**: se basa en pronósticos climáticos reales.
> - Es **adaptable**: puede actualizarse cada año con nuevos datos.
> - Es **interpretable**: gracias a la lógica difusa, podemos explicar por qué se recomienda cierta fecha."

---

### DIAPOSITIVA 29: ALTERNATIVA PSO
**[CRISTIAN]**

> "Como alternativa al algoritmo genético, también implementé el algoritmo de **Enjambre de Partículas** o PSO.
>
> **¿Qué es PSO?** Es un algoritmo inspirado en el comportamiento de bandadas de aves o cardúmenes. Cada 'partícula' representa una fecha candidata y se mueven por el espacio de búsqueda, influenciadas por su mejor posición histórica y la mejor del grupo.
>
> Las **ventajas de PSO** sobre el algoritmo genético son:
> - Tiene menos parámetros de configuración.
> - La convergencia es más suave y predecible.
> - No requiere operadores de cruce ni mutación.
>
> Ambos algoritmos llegan a resultados similares, lo cual valida que el óptimo encontrado es robusto."

---

### DIAPOSITIVA 30: CÓDIGO PSO
**[CRISTIAN]**

> "El código del PSO usa la librería Mealpy. La función objetivo es prácticamente idéntica al fitness del algoritmo genético: valida restricciones, obtiene el clima, y suma la aptitud de cada día.
>
> La configuración usa 50 épocas y 20 partículas. El resultado se obtiene llamando a `modelo.solve()`."

---

### DIAPOSITIVA 33: CONCLUSIONES
**[CRISTIAN]**

> "Para cerrar, nuestros **logros técnicos** fueron:
>
> - Implementamos exitosamente un sistema híbrido que integra **3 técnicas de IA**.
> - La **Red LSTM** logra predicciones climáticas para todo el año.
> - El **Sistema Difuso** traduce condiciones climáticas en aptitud de siembra.
> - El **Algoritmo Genético** (y PSO) encuentran eficientemente la fecha óptima.
>
> El **impacto potencial** de este sistema incluye:
> - Ser una herramienta de apoyo real para agricultores de la Mixteca.
> - Contribuir a la reducción del riesgo de pérdida de cosechas.
> - Ser un modelo replicable para otras regiones agrícolas de México."

---

### DIAPOSITIVA 34: GRACIAS
**[CRISTIAN / TODOS]**

> "Con esto concluimos nuestra presentación. Agradecemos su atención.
>
> ¿Tienen alguna pregunta?"

---

## NOTAS PARA LA PRESENTACIÓN

1. **Tiempo sugerido por sección:**
   - Ossiel (Intro + Fase 0): ~8 minutos
   - Aneli (Fase 1): ~5 minutos
   - Ramón (Fase 2): ~7 minutos
   - Cristian (Fase 3 + Cierre): ~10 minutos

2. **Transiciones recomendadas:**
   - Cada integrante debe cerrar diciendo "le paso la palabra a [nombre]".
   - Mantener contacto visual con la audiencia.
   - Usar la gráficas para señalar puntos específicos.

3. **Posibles preguntas del jurado:**
   - ¿Por qué eligieron 120 días como ciclo de cultivo?
   - ¿Qué tan precisa es la predicción de la LSTM?
   - ¿Por qué usaron 9 reglas difusas y no más?
   - ¿Cuál es la fecha óptima que encontró el sistema?
   - ¿Se podría aplicar a otros cultivos?

4. **Respuestas sugeridas:**
   - El ciclo de 120 días es el período típico de maduración del maíz en la región.
   - La LSTM tiene un error promedio de ±2-3°C en temperatura y ±5mm en precipitación.
   - Las 9 reglas cubren todas las combinaciones de 3 niveles × 3 niveles.
   - La fecha típica encontrada está entre el día 140-160 (mediados de mayo a inicios de junio).
   - Sí, ajustando las reglas difusas y el ciclo de cultivo se puede adaptar a otros cultivos.

---

**¡Éxito en su presentación!** 🌽
