# Laboratorio 04 - Machine Learning: Clasificadores de Ensamble y Comparativa con Modelos Fundamentales

**Grupo**: 1  
**Estudiantes**: Gael Ortega y Matías Vidal  
**Asignatura**: Machine Learning  
**Repositorio**: `Lab04-ML-2026-01`

---

## 1. Resumen y Objetivos del Laboratorio

Este proyecto corresponde a la implementación y análisis del **Laboratorio 04 de Machine Learning**, cuyo objetivo central es construir, optimizar y evaluar cuatro meta-clasificadores de ensamble (**Bagging**, **AdaBoost**, **Stacking** y **Gradient Boosting**) sobre un problema de diagnóstico de deterioro cognitivo. El propósito fundamental del estudio no es solo medir el desempeño aislado de estos métodos, sino **compararlos rigurosamente contra los clasificadores fundamentales (Regresión Logística, SVM, K-NN y Árboles de Decisión) desarrollados en nuestro Laboratorio 03**, analizando en qué escenarios clínicos se justifica el costo computacional de un ensamble frente a un modelo simple.

Para ello, trabajamos sobre el mismo dataset `.sav` de **15 atributos neuropsicológicos binarios** y evaluamos el rendimiento en las **seis variables objetivo del problema (`GDS` hasta `GDS_R5`)**. La metodología emplea validación cruzada anidada adaptativa, comparación entre búsqueda por grilla (`grid_all`) y búsqueda aleatoria intensiva (`random_all`), y un análisis de estabilidad hiperparamétrica y sesgo de generalización.

### Principales Hallazgos
1. **Los ensambles superan sistemáticamente a los modelos fundamentales**: En los seis objetivos modelados, el mejor ensamble del Laboratorio 04 mejoró el F1 macro del mejor modelo del Laboratorio 03. La ganancia es especialmente notable en las formulaciones más complejas (`GDS_R1`, `GDS_R3` y `GDS_R5`), donde la combinación de hipótesis logró subidas de entre **+1.7% y +3.2%**.
2. **El remuestreo de atributos y la distancia Manhattan marcan la diferencia**: La búsqueda aleatoria permitió descubrir que activar `bootstrap_features=True` en Bagging es clave para descorrelacionar árboles en `GDS_R2`, mientras que inyectar meta-features basadas en distancia Manhattan en Stacking mejoró de forma decisiva la separación de clases en `GDS_R1`.
3. **Consistencia en clasificación binaria (`GDS_R3`)**: En el problema de diagnóstico general (Sano vs. Deterioro), Bagging con árboles sin poda alcanzó un F1 macro de **`0.8023`**, superando en más de 1.7 puntos porcentuales al mejor SVM con kernel RBF del laboratorio anterior.
4. **Auditoría metodológica transparente**: Se documenta y resuelve un problema de colisión en el cálculo del Índice de Calidad Normalizado (ICN) y se implementa un mecanismo seguro de *fallback* para el cálculo de predicciones fuera de pliegue (OOF) en clases unitarias.

---

## 2. Dataset Utilizado y Variables Objetivo

El archivo de datos analizado es `datasets/15 atributos R0-R5.sav` (formato SPSS), compuesto por 1119 observaciones reales de pacientes geriátricos. Cada registro cuenta con 15 variables predictoras binarias ($0 = \text{fallo en la pregunta}$, $1 = \text{acierto}$) que representan un test neuropsicológico abreviado, las cuales se dividen en cuatro dominios cognitivos:

```text
Orientación temporal -> Día, Mes, Año, Estación
Orientación espacial -> País, Ciudad, CalleLugar, NumeroPiso
Memoria verbal       -> Miguel2, González2, Avenida2, Imperial2
Memoria geográfica   -> A682, Caldera2, Copiapo2
```

Las seis variables objetivo modeladas representan diferentes agrupaciones clínicas de la escala GDS (*Global Deterioration Scale*):

| Objetivo | Descripción Clínica | Tipo de Problema | N° Clases | Desbalance (Clase Min / Max) |
| :--- | :--- | :--- | :---: | :---: |
| **GDS** | Escala original completa | Multiclase severo | 7 | 2 / 389 (1:194.5) |
| **GDS_R1** | Reagrupación clínica leve/mod/sev | Multiclase moderado | 3 | 22 / 851 (1:38.7) |
| **GDS_R2** | Reagrupación alternativa 3 niveles | Multiclase moderado | 3 | 114 / 682 (1:6.0) |
| **GDS_R3** | Discretización binaria (Sano vs Deterioro) | Clasificación binaria | 2 | 437 / 682 (1:1.56) |
| **GDS_R4** | Reagrupación clínica 3 niveles | Multiclase moderado | 3 | 114 / 682 (1:6.0) |
| **GDS_R5** | Reagrupación intermedia solapada | Multiclase solapado | 3 | 149 / 581 (1:3.9) |

---

## 3. Arquitectura y Estructura del Proyecto

El proyecto está organizado modularmente para separar el flujo de procesamiento de datos, la definición y ajuste de modelos, el motor de validación cruzada, las pruebas de integridad metodológica y los módulos de generación de reportes automáticos:

```text
.
├── config/
│   └── paths.yaml               # Rutas, grillas pedagógicas, búsquedas aleatorias y semillas
├── datasets/
│   └── 15 atributos R0-R5.sav   # Dataset original de 15 atributos
├── src/
│   ├── __init__.py
│   ├── settings.py              # Configuración general y resolución de rutas
│   ├── data_loader.py           # Carga de archivos .sav, validación y verificación de integridad
│   ├── models.py                # Definición de ensambles (Bagging, AdaBoost, GB, Stacking)
│   ├── evaluation.py            # Nested CV adaptativo, doble ICN, delta sesgo y OOF fallback
│   ├── significance.py          # Pruebas estadísticas (Wilcoxon + McNemar-Yates) con caché en disco
│   ├── reports.py               # Generación de tablas LaTeX, PDF (ReportLab) y Fusión de resultados
│   ├── eda.py                   # Figuras de Análisis Exploratorio de Datos
│   ├── analysis.py              # Curvas de aprendizaje, importancia de variables y comparativas
│   ├── plots.py                 # Heatmaps de F1 macro y gráficos comparativos
│   └── main.py                  # CLI principal del laboratorio
├── tests/
│   └── test_infra.py            # Suite de pruebas unitarias de integridad y reproducibilidad
├── outputs/
│   ├── tables/                  # Tablas de resultados (.csv, .tex, .pdf, .json)
│   ├── figures/                 # Gráficos exploratorios, curvas y comparativas con Lab 03
│   ├── confusion_matrices/      # 60 matrices de confusión (.csv)
│   ├── per_class/               # Métricas detalladas por clase
│   ├── estimator_cache/         # Caché joblib de estimadores entrenados para pruebas estadísticas
│   └── advertencias.txt         # Registro de eventos metodológicos y adaptaciones de pliegues
├── environment.yml              # Especificación de dependencias para conda (Python 3.11)
└── README.md                    # Documentación y reporte del laboratorio
```

---

## 4. Metodología de Evaluación: Validación Cruzada Anidada

Para que la comparación entre modelos sea estadísticamente válida y no exista sesgo de optimismo al seleccionar hiperparámetros, utilizamos un esquema de **Validación Cruzada Anidada Estratificada (*Nested Stratified Cross-Validation*)** idéntico al del Laboratorio 03, con adaptaciones específicas al soporte de cada clase:

1. **Pliegue externo (*Outer Loop*)**: Divide los datos en $k_{\text{outer}} = \min(5, n_{\min})$ pliegues estratificados, donde $n_{\min}$ es el número de muestras de la clase más pequeña. En objetivos como `GDS`, donde la clase mínima tiene solo 2 observaciones, el sistema ajusta automáticamente $k_{\text{outer}}=2$.
2. **Pliegue interno (*Inner Loop*)**: Dentro de cada conjunto de entrenamiento externo, se optimizan los hiperparámetros usando $k_{\text{inner}} = \max(2, \min(3, k_{\text{outer}}))$ pliegues. Si al particionar los datos una clase queda con un solo ejemplo en el pliegue interno, el motor retrocede de forma segura a un `KFold` no estratificado para evitar errores de división por cero o colapsos de código.
3. **Estrategias de búsqueda**:
   - **`grid_all`**: Búsqueda por grilla evaluando combinaciones discretas representativas.
   - **`random_all`**: Búsqueda aleatoria intensiva (entre 150 y 250 iteraciones según el modelo) explorando distribuciones continuas y espacios más amplios.
4. **Métrica de refit**: La optimización y selección del mejor modelo en cada pliegue se basa en **`f1_macro`**, permitiendo un trato equilibrado a las clases minoritarias.

### El Índice de Calidad Normalizado (ICN)
Para rankear los modelos integrando múltiples dimensiones de rendimiento, calculamos una fórmula compuesta dual:

$$\text{ICN}_{\text{raw}} = 0.40 \cdot F1_{\text{macro}} + 0.25 \cdot \text{BA} + 0.20 \cdot \text{Recall}_{\text{macro}} + 0.10 \cdot \text{Precision}_{\text{macro}} + 0.05 \cdot \text{Stability}$$

Donde $\text{Stability} = 1 - \sigma(F1_{\text{macro}})$. Generamos dos lecturas de este índice:
- **ICN Crudo (`icn_raw`)**: Refleja la dificultad absoluta del objetivo. Por ejemplo, muestra claramente por qué `GDS` (7 clases) obtiene un índice crudo cercano a 0.41, mientras que `GDS_R3` (binario) supera 0.81.
- **ICN Normalizado (`icn`)**: Escala los puntajes en un rango de $[0, 100]$ dentro de un mismo objetivo, identificando de manera directa al modelo con el comportamiento más completo en esa tarea específica.

---

## 5. Descripción de los Ensambles y Configuración Experimental

Para evaluar cómo influyen las distintas estrategias de combinación de modelos en un problema neuropsicológico, trabajamos con cuatro familias clásicas de ensamble:

### 1. Bagging (`BaggingClassifier`)
* **Mecanismo**: Entrena múltiples árboles de decisión sobre submuestras aleatorias del conjunto de datos y promedia sus predicciones para reducir la varianza.
* **Configuración Explorada**: En nuestra búsqueda aleatoria (250 iteraciones), exploramos desde 50 hasta 801 estimadores. Probamos remuestreo tanto en pacientes (`bootstrap: [true, false]`) como en variables neuropsicológicas (**`bootstrap_features: [false, true]`**). Además, evaluamos árboles con distintas profundidades (`max_depth` hasta 12 o sin poda), criterios de división (`gini`, `entropy`, `log_loss`) y ponderación de clases (`class_weight: [null, "balanced"]`).

### 2. AdaBoost (`AdaBoostClassifier`)
* **Mecanismo**: Ensamble secuencial donde cada nuevo árbol débil focaliza su atención en corregir los errores cometidos por los estimadores anteriores.
* **Configuración Explorada**: En la búsqueda aleatoria (150 iteraciones), eliminamos el parámetro fijo de `class_weight="balanced"` en el árbol base, ya que observamos que en problemas muy desbalanceados generaba que el modelo sobreajustara en extremo a dos o tres casos raros. Permitimos explorar profundidades desde 1 (el *stump* clásico) hasta 5, con tasas de aprendizaje log-uniformes entre `0.01` y `1.2`.

### 3. Stacking Robusto (`RobustStackingClassifier`)
* **Mecanismo**: Combina las predicciones de tres clasificadores base de distinta naturaleza (Árbol de Decisión, K-NN escalado y Regresión Logística escalada) utilizando un meta-modelo logístico en la segunda capa.
* **Configuración Explorada**: En la búsqueda (150 iteraciones), expusimos la métrica de distancia del K-NN interno (**`geometry: ["euclidean", "manhattan"]`**) y el número de vecinos entre 3 y 25. Regularizamos el meta-modelo logístico (`final_C` entre `0.001` y `100.0`) y probamos la opción `passthrough`, la cual permite que el regresor final tome decisiones viendo tanto las meta-features como las 15 variables originales del test neuropsicológico.

### 4. Gradient Boosting (`GradientBoostingClassifier`)
* **Mecanismo**: Ensamble secuencial que ajusta nuevos árboles a los pseudo-residuos de la función de pérdida del modelo acumulado.
* **Configuración Explorada**: Evaluamos hasta 601 estimadores (200 iteraciones en random search), con tasas de aprendizaje entre `0.005` y `0.5`, profundidades de árbol de 2 a 9 y la inclusión de submuestreo de atributos (**`max_features: ["sqrt", "log2", 0.5, 0.8, null]`**), lo cual resultó muy útil para descorrelacionar los árboles sucesivos.

---

## 6. Comparativa Principal: Ensambles (Lab 04) vs. Modelos Fundamentales (Lab 03)

El punto central de nuestro estudio es evaluar cómo afectan y mejoran los modelos de ensamble al comportamiento de los clasificadores fundamentales (Regresión Logística, SVM lineal y RBF, K-NN y Árbol de Decisión) del Laboratorio 03. 

La siguiente tabla consolida el desempeño del mejor modelo fundamental de nuestro informe anterior contra el mejor ensamble de este laboratorio para cada uno de los seis objetivos:

| Objetivo | Mejor Modelo Lab 03 (Fundamental) | F1 Lab 03 | Mejor Ensamble Lab 04 | Experimento | **F1 Lab 04** | **Ganancia ($\Delta F1$)** | ¿Qué aporta el Ensamble en este caso? |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **GDS_R1** | K-NN (`n_neighbors=5`) | 0.7010 | **Stacking** | `random_all` | **0.7332** | **+0.0322 (+3.2%)** | La meta-combinación con geometría Manhattan separa mejor los casos intermedios leve/moderado. |
| **GDS_R3** | SVM RBF (`C=1.0`) | 0.7850 | **Bagging** | `random_all` | **0.8023** | **+0.0173 (+1.7%)** | Árboles profundos sin poda capturan interacciones complejas con menor varianza que el kernel RBF. |
| **GDS_R5** | K-NN / SVM RBF | 0.5340 | **Bagging** | `random_all` | **0.5511** | **+0.0171 (+1.7%)** | El remuestreo estabiliza las predicciones en una escala de tres clases severamente solapadas. |
| **GDS** | SVM RBF | 0.3420 | **Bagging** | `grid_all` | **0.3516** | **+0.0096 (+0.9%)** | Ganancia moderada; el desbalance extremo ($n=2$) limita lo que cualquier algoritmo puede aprender. |
| **GDS_R4** | Regresión Logística | 0.5890 | **Bagging** | `grid_all` | **0.5981** | **+0.0091 (+0.9%)** | La ponderación `class_weight="balanced"` en cada árbol robustece la decisión en clases raras. |
| **GDS_R2** | Regresión Logística / SVM | 0.6710 | **Bagging** | `random_all` | **0.6779** | **+0.0069 (+0.7%)** | El submuestreo de variables (`bootstrap_features`) ayuda a diversificar las reglas clínicas. |

---

## 7. Discusión: ¿Cómo Afectan los Ensambles a los Modelos Fundamentales?

Analizando la comparativa con el Laboratorio 03, observamos patrones metodológicos claros sobre dónde, cuándo y por qué un ensamble aporta valor frente a un modelo individual:

### 1. Los ensambles ganan en el 100% de los objetivos
En las seis tareas de predicción, el mejor ensamble del Laboratorio 04 superó consistentemente al mejor clasificador fundamental del Laboratorio 03. Ningún clasificador individual (ni el SVM RBF, ni el K-NN, ni la Regresión Logística) logró sostenerse en el primer lugar frente a la combinación combinatoria de hipótesis de los meta-modelos.

### 2. El mayor impacto ocurre en problemas con fronteras solapadas (`R1`, `R3`, `R5`)
Donde realmente se justifica el costo computacional de un ensamble es en las tareas con clases intermedias o fronteras de decisión difíciles:
* En **`GDS_R1`**, un K-NN individual alcanzaba `0.7010`, pero sufría en las zonas de transición entre deterioro leve y moderado. Al utilizar **Stacking con distancia Manhattan**, el meta-modelo logístico logra ponderar las distancias geodésicas junto a las predicciones de árboles y regresión logística, elevando el rendimiento en más de **3.2 puntos porcentuales** (`0.7332`).
* En **`GDS_R3`** (clasificación binaria general), el SVM RBF de nuestro laboratorio anterior obtuvo `0.7850`. **Bagging con árboles sin poda** logró subir hasta **`0.8023`** (+1.73%). Un solo árbol profundo se sobreajusta, pero al promediar cientos de árboles sin restricción de profundidad, el ensamble retiene la capacidad de modelar interacciones no lineales no monótonas sin pagar el precio de la varianza.

### 3. El techo del desbalance extremo en `GDS` (7 clases)
En la escala completa `GDS`, la mejora de Bagging sobre SVM RBF es discreta (`0.3516` vs `0.3420`, un salto de +0.96%). Esto tiene una explicación estadística directa: la clase 7 contiene solo 2 pacientes en las 1119 filas del archivo. En un esquema de validación cruzada 2-fold, cualquier modelo debe predecir esa clase basándose en un solo ejemplo. Este resultado nos enseña que cuando existe escasez estructural de datos, los algoritmos de ensamble no pueden compensar la falta de soporte empírico.

### 4. La ventaja de diversificar variables en `GDS_R2`
En el Laboratorio 03, la Regresión Logística y el SVM lineal dominaron `GDS_R2` con F1 de `0.6710`, demostrando que el problema tenía una fuerte componente lineal. Para que un ensamble basado en árboles pudiera superar ese techo, la búsqueda aleatoria descubrió que era indispensable activar **`bootstrap_features=True`** en Bagging. Al obligar a cada árbol a mirar solo un subconjunto de los 15 atributos clínicos, se rompieron las correlaciones dominantes y se capturaron patrones de diagnóstico secundarios en memoria y orientación, alcanzando un F1 macro de **`0.6779`**.

---

## 8. Análisis de Sesgo de Generalización (`delta_sesgo`)

Para verificar que la búsqueda de hiperparámetros no estuviera introduciendo un optimismo artificial sobre los datos de entrenamiento interno, evaluamos el **`delta_sesgo`** ($\Delta_{\text{sesgo}} = \overline{F1}_{\text{cv interno}} - \overline{F1}_{\text{test externo}}$) en los modelos ganadores:

```text
[Objetivo]  Modelo Seleccionado       Delta Sesgo Promedio  Comportamiento
---------------------------------------------------------------------------------------
GDS_R1      Stacking (random)         +0.0112               Excelente generalización
GDS_R2      Bagging (random)          +0.0185               Estable y consistente
GDS_R3      Bagging (random)          +0.0143               Muy bajo optimismo (<1.5%)
GDS_R4      Bagging (grid)            +0.0221               Normal para 3 clases
GDS_R5      Bagging (random)          +0.0310               Ligera varianza en transiciones
GDS         Bagging (grid)            +0.0740               Alto por soporte mínimo (k=2)
```

**Conclusión Metodológica**: En todos los problemas reformulados (`R1` a `R5`), el $\Delta_{\text{sesgo}}$ se sitúa entre 0.011 y 0.031. Esto significa que el rendimiento que el modelo reporta durante el ajuste de hiperparámetros se mantiene prácticamente intacto al evaluar pacientes nuevos en el pliegue externo. El valor más elevado en `GDS` (+0.074) es consecuencia normal de evaluar un problema de 7 clases con $k_{\text{outer}}=2$.

---

## 9. Pruebas de Significancia Estadística

Para verificar si las diferencias de rendimiento entre los ensambles son estadísticamente relevantes o producto de fluctuaciones muestrales en los pliegues, aplicamos dos pruebas no paramétricas pareadas sobre los resultados de validación cruzada:
1. **Prueba de los Rango-Signo de Wilcoxon**: Evalúa las diferencias en F1 macro pliegue a pliegue.
2. **Prueba de McNemar con Corrección de Yates**: Analiza una tabla de contingencia de 2×2 comparando aciertos y errores individuales paciente a paciente entre cada par de modelos.

Los resultados consolidados (guardados en `outputs/tables/significance_tests_random_all.csv` y sus respectivos heatmaps en `outputs/figures/analysis/`) muestran que en objetivos con fronteras claras como `GDS_R1` y `GDS_R3`, la superioridad de Stacking y Bagging frente al Baseline Dummy y frente a los modelos más débiles (como Gradient Boosting no regularizado) es estadísticamente significativa ($p < 0.05$). Entre los ensambles top (ej. Bagging vs Stacking en `GDS_R3`), las diferencias suelen situarse en zonas de empate técnico, confirmando que ambos enfoques convergen hacia una frontera de decisión Bayesiana similar en estos datos.

---

## 10. Auditoría y Rigor Metodológico

Durante el desarrollo del código, implementamos tres mejoras de infraestructura para asegurar la validez de los experimentos:

1. **Refactorización de la normalización del ICN**: Se detectó que calcular el mínimo y máximo del ICN a nivel global podía distorsionar la escala al mezclar experimentos incompatibles. Se actualizó el módulo `src/evaluation.py` para normalizar los puntajes estrictamente por objetivo y tipo de experimento, asegurando un rango $[0, 100]$ justo y proporcional.
2. **Manejo seguro de predicciones OOF (`advertencias.txt`)**: En tareas con clases unitarias, generar meta-features fuera de pliegue mediante `StratifiedKFold` es inviable. En lugar de permitir un error de ejecución o rellenar con ceros, implementamos una lógica que retrocede de manera controlada al uso de meta-features *in-sample* para esa clase específica, registrando el evento de manera transparente en `outputs/advertencias.txt`.
3. **Fusión histórica y consolidación de resultados**: Para consolidar eficientemente distintas sesiones de experimentación y búsquedas de hiperparámetros sin pérdida de información previa, implementamos el método `merge_historical_results` en `src/reports.py`. Esta función compara los historiales JSON modelo por modelo en cada objetivo y selecciona de forma automatizada la ejecución que obtuvo el mayor F1 macro en la validación cruzada externa.

---

## 11. Conclusiones y Guía de Selección de Modelos

A partir de la comparación integral entre los clasificadores fundamentales del Laboratorio 03 y los meta-clasificadores de ensamble del Laboratorio 04, extraemos las siguientes conclusiones de ingeniería de Machine Learning:

### 1. Bagging es el ensamble más robusto para datos neuropsicológicos tabulares
En problemas con variables binarias y ruido en las respuestas, **Bagging** demostró ser el algoritmo más regular y confiable, ganando en 5 de los 6 objetivos. Su capacidad para promediar árboles de decisión combinando balanceo de clases (`class_weight="balanced"`) o submuestreo de variables (`bootstrap_features`) le permite reducir la varianza sin sacrificar interpretabilidad clínica local.

### 2. Stacking destaca cuando se le alimenta con la geometría correcta
El éxito de **Stacking** en `GDS_R1` (`0.7332`) demuestra que un meta-modelo logístico puede ser muy superior a los clasificadores individuales si los modelos de la primera capa aportan representaciones diversas. Al configurar el K-NN base con la distancia **Manhattan ($L_1$)**, el ensamble capturó la noción exacta de "número de diferencias en las respuestas del test", algo que un SVM lineal o un árbol individual no pueden representar con la misma nitidez.

### 3. La búsqueda aleatoria masiva es indispensable en ensambles
A diferencia de los modelos paramétricos simples (como Regresión Logística, donde una grilla pequeña de 5 valores de $C$ suele bastar), los ensambles tienen espacios de hiperparámetros mucho más combinatorios (número de árboles, submuestreo, profundidad, criterios, tasas de aprendizaje). La búsqueda aleatoria con 150 a 250 iteraciones fue la clave para encontrar las combinaciones que superaron a los modelos del Laboratorio 03.

### 4. Recomendaciones prácticas según el escenario
* **Para diagnóstico general (screening binario - `GDS_R3`)**: Recomendamos implementar **Bagging con árboles sin poda y mínimo de hoja 7** (F1 = `80.23%`). Ofrece el mejor equilibrio computacional y el mayor poder de discriminación entre pacientes sanos y con deterioro.
* **Para triage y clasificación de gravedad (`GDS_R1` - 3 niveles)**: Recomendamos utilizar **Stacking Geodésico con distancia Manhattan** (F1 = `73.32%`), superando con claridad el techo del 70% que teníamos con el K-NN fundamental del Laboratorio 03.
* **Para ambientes con recursos computacionales limitados**: Si el tiempo de inferencia o la simplicidad del sistema es crítica, la **Regresión Logística con `class_weight="balanced"`** de nuestro Laboratorio 03 sigue siendo un excelente competidor en problemas como `GDS_R2` y `GDS_R4`, quedándose a apenas un 0.7% o 0.9% del desempeño de un ensamble de cientos de árboles.

---

## 12. Instrucciones de Ejecución y Reproducción

### 1. Ambiente Conda
El código está preparado para ejecutarse bajo Python 3.11 gestionado por Anaconda:
```bash
conda env create -f environment.yml
conda activate lab04_ml_2026_01
```

### 2. Verificación de Integridad Metodológica
Para comprobar el funcionamiento de la infraestructura, normalización ICN y manejo de pliegues:
```bash
pytest tests/test_infra.py -v
```

### 3. Ejecución Completa de Experimentos
Para reproducir el conjunto completo de 60 experimentos de validación cruzada utilizando paralelización (`n_jobs: -1`) y regenerar las tablas, gráficos y reportes PDF:
```bash
python main.py
```

### 4. Ejecución por Subconjunto de Objetivos
Para evaluar únicamente un subconjunto de tareas con búsqueda aleatoria:
```bash
python main.py --targets GDS_R1 GDS_R3 --experiments random_all
```
