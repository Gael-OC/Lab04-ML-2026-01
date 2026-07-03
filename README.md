# Lab04-ML-2026-01

Implementacion del Laboratorio 04: clasificadores de ensamble (Bagging, AdaBoost, Stacking y Gradient Boosting) sobre el dataset `.sav` de 15 atributos binarios. Compara dos estrategias de busqueda de hiperparametros (`grid_all` con `GridSearchCV` y `random_all` con `RandomizedSearchCV` con distribuciones reales), mide el doble ICN (crudo comparable entre objetivos, normalizado comparable dentro de un objetivo), reporta `delta_sesgo`, significancia estadistica (Wilcoxon + McNemar-Yates con correccion de Yates) y contrasta los ensambles con los clasificadores simples del Laboratorio 03.

## 1. Que hay dentro

- `src/models.py`: `DummyClassifier`, `BaggingClassifier` (con `class_weight` configurable en el grid), `AdaBoostClassifier` (stump por defecto), `GradientBoostingClassifier` y `RobustStackingClassifier` (con soporte para `passthrough`, OOF cuando `n_min >= 2` e in-sample si no).
- `src/evaluation.py`: validacion cruzada anidada unificada, distribuciones materializadas desde YAML (`randint`, `uniform`, `loguniform`, `values`), ICN doble, `delta_sesgo`, advertencias automaticas, `--keep-estimators` con cache en disco.
- `src/main.py`: CLI con `--targets`, `--models`, `--experiments`, `--keep-estimators`, `--skip-eda`, `--skip-analysis`, `--skip-plots`.
- `src/reports.py`: CSV con 22 columnas, LaTeX con `booktabs`, PDF con `reportlab` y figuras incrustadas.
- `src/analysis.py`: learning curves, feature importance por ensamble, estabilidad de hiperparametros, zero-recall, low-support, per-fold stability, bootstrap CI al 95%, visualizacion del primer arbol de Gradient Boosting, tests de significancia (Wilcoxon + McNemar-Yates), comparativa `grid_all` vs `random_all`, comparativa **Lab 3 vs Lab 4**.
- `src/significance.py`: cache de estimadores y tests pareados con carga desde disco.
- `src/eda.py`: 4 figuras estandar (class distribution, feature correlation, target relationships, eda summary).
- `src/plots.py`: heatmap F1 macro.
- `config/paths.yaml`: rutas, semillas centralizadas, grids exhaustivas y distribuciones aleatorias.
- `environment.yml`: dependencias (Python 3.11, scikit-learn, scipy, reportlab, etc.).

## 2. Idea central

Cada modelo se evalua con el mismo protocolo de **validacion cruzada anidada**:

1. Se toma un objetivo (`GDS`, `GDS_R1`, ..., `GDS_R5`).
2. Se separa en `k_outer = min(5, n_min)` folds externos estratificados.
3. Dentro de cada fold externo se hace `GridSearchCV` o `RandomizedSearchCV` con `f1_macro` y un `k_inner = min(3, k_outer)` ajustado a la clase minoritaria.
4. El mejor estimador interno se evalua sobre el fold externo.
5. Al final se reportan promedios, desviaciones, ICN doble, `delta_sesgo`, matrices de confusion, metricas por clase y warnings.

Esto se aplica sobre los **6 objetivos × 5 modelos × 2 experimentos = 60 corridas**.

## 3. Ambiente conda

```bash
conda env create -f environment.yml
conda activate lab04_ml_2026_01
```

El dataset debe estar en `datasets/15 atributos R0-R5.sav`. Si cambia de ubicacion, ajustar `dataset.path` en `config/paths.yaml`.

## 4. Ejecucion

Corrida completa con cache de estimadores (necesario para tests de significancia):

```bash
python main.py --keep-estimators
```

Subconjuntos:

```bash
python main.py --targets GDS_R3 GDS_R5
python main.py --models adaboost gradient_boosting
python main.py --experiments random_all
python main.py --experiments grid_all --targets GDS_R3
```

Saltar partes:

```bash
python main.py --skip-eda
python main.py --skip-analysis
python main.py --skip-plots
```

## 5. Configuracion

`config/paths.yaml` controla:

- `dataset.path`: ruta al `.sav`.
- `outputs`: carpetas de salida.
- `experiment.targets`: lista de los seis objetivos.
- `experiment.experiments`: orden de las corridas (`grid_all` primero, `random_all` despues).
- `experiment.validation`: `max_outer_folds`, `max_inner_folds`, `scoring`, `n_jobs`.
- `experiment.random_state`: `global`, `outer_cv`, `inner_cv`, `model`, `search` (semillas centralizadas).
- `hyperparameter_search.grid.models.<modelo>.params`: grilla exhaustiva para `GridSearchCV`.
- `hyperparameter_search.random.models.<modelo>.distributions`: distribuciones para `RandomizedSearchCV`.

Distribuciones soportadas:

- `randint` (enteros), con `low` y `high` exclusivos.
- `uniform` (reales), con `low` y `high`.
- `loguniform` (reales en escala log), con `low` y `high`.
- `values` (lista discreta).

## 6. Hiperparametros por modelo

### Dummy

Sin hiperparametros. Sirve como cota inferior.

### Bagging

- Grid: `n_estimators ∈ {50, 100, 200}`, `max_samples ∈ {0.7, 1.0}`, `max_features ∈ {0.5, 1.0}`, `max_depth ∈ {3, 5, null}`, `min_samples_leaf ∈ {1, 5}`, `class_weight ∈ {null, "balanced"}` (144 combinaciones).
- Random: `n_estimators ~ randint(50, 501)`, `max_samples ~ uniform(0.5, 1.0)`, `max_features ~ uniform(0.5, 1.0)`, `max_depth ∈ {1, 2, 3, 5, null}`, `min_samples_leaf ~ randint(1, 6)`, `class_weight ∈ {null, "balanced"}` (40 extracciones).

### AdaBoost

- Grid: `n_estimators ∈ {50, 100, 200}`, `learning_rate ∈ {0.05, 0.1, 0.5, 1.0}`, `max_depth ∈ {1, 2, 3}` (36 combinaciones).
- Random: `n_estimators ~ randint(50, 401)`, `learning_rate ~ loguniform(0.05, 1.5)`, `max_depth ∈ {1, 2, 3}` (40 extracciones).

### Stacking

Bases: `DecisionTreeClassifier` (con `class_weight="balanced"`), K-NN escalado, regresion logistica escalada. Meta: regresion logistica con `class_weight="balanced"`.

- Grid: `final_C ∈ {0.1, 1.0, 10.0}`, `logistic_C ∈ {0.1, 1.0, 10.0}`, `tree_max_depth ∈ {3, null}`, `n_neighbors ∈ {3, 7, 11}`, `passthrough ∈ {false, true}` (108 combinaciones).
- Random: `final_C ~ loguniform(0.01, 10)`, `logistic_C ~ loguniform(0.01, 10)`, `tree_max_depth ∈ {2, 3, null}`, `n_neighbors ~ randint(3, 16)`, `passthrough ∈ {false, true}` (40 extracciones).

### Gradient Boosting

- Grid: `n_estimators ∈ {50, 100, 200}`, `learning_rate ∈ {0.05, 0.1, 0.2}`, `max_depth ∈ {2, 3}`, `subsample ∈ {0.7, 1.0}`, `min_samples_leaf ∈ {1, 5}` (72 combinaciones).
- Random: `n_estimators ~ randint(100, 501)`, `learning_rate ~ loguniform(0.01, 0.3)`, `max_depth ~ randint(2, 6)`, `subsample ~ uniform(0.6, 1.0)`, `min_samples_leaf ~ randint(1, 6)` (40 extracciones).

## 7. Metricas y seleccion

- Metrica principal de busqueda: `f1_macro` (configurable via `validation.scoring`).
- Metricas reportadas: `accuracy`, `balanced_accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `stability` (1 - std F1), `icn_raw` (suma ponderada 0.40·F1 + 0.25·BA + 0.20·recall + 0.10·precision + 0.05·stability) e `icn` (min-max normalizado dentro de target).
- `delta_sesgo` = `best_score_internal_mean` - `f1_macro_mean` (optimismo del CV interno).
- `advertencias.txt` registra automaticamente: k_inner ajustado por soporte minimo, k interno no estratificado cuando una clase tiene 1 ejemplo en entrenamiento externo, y fits fallidos de AdaBoost en grids/random con configuraciones degeneradas.

## 8. Salidas

```
outputs/
├── tables/
│   ├── resumen_resultados.csv          # 22 columnas, 60 filas
│   ├── resultados_detallados.json
│   ├── resultados_experimentos.tex     # LaTeX con booktabs
│   ├── resultados_experimentos.pdf     # PDF con reportlab + figuras
│   ├── distribucion_clases.csv
│   ├── lab3_vs_lab4_comparison.csv
│   ├── grid_vs_random_summary.csv
│   ├── bootstrap_ci_95.csv
│   ├── significance_tests_grid_all.csv
│   ├── significance_tests_random_all.csv
│   ├── zero_recall_classes.csv
│   └── low_support_classes.csv
├── confusion_matrices/                 # 60 archivos CSV (target × modelo × experimento)
├── per_class/                          # 60 archivos CSV de metricas por clase
├── estimator_cache/                    # 60 archivos joblib (1 por (target, modelo, experimento))
├── figures/
│   ├── eda/                            # 4 figuras EDA
│   ├── experiments/                    # heatmap F1 macro
│   └── analysis/                       # learning curves, importance, hyperparam stability,
│                                       # zero-recall, per-fold, bootstrap forest, gb trees,
│                                       # significance heatmaps, grid_vs_random, lab3_vs_lab4
└── advertencias.txt
```

## 9. Comparativa con Lab 3

`outputs/tables/lab3_vs_lab4_comparison.csv` lee `Lab 3 Anterior/Lab03-ML-2026-01/outputs/tables/resumen_resultados.csv` y reporta, para cada objetivo, el mejor F1 macro de Lab 3 y el mejor F1 macro entre los 4 ensambles de Lab 4, junto con la diferencia absoluta y relativa.

`outputs/figures/analysis/lab3_vs_lab4_f1.png` muestra la misma comparativa como barras agrupadas con desviación estándar.

`outputs/figures/analysis/grid_vs_random_heatmap.png` y `grid_vs_random_diff_heatmap.png` muestran el F1 macro por (objetivo, modelo) en cada experimento y la diferencia `random_all - grid_all`.

## 10. Validacion cruzada anidada y semilla

`experiment.random_state`:

- `global` (42): respaldo.
- `outer_cv` (42): semilla del `StratifiedKFold` externo.
- `inner_cv` (123): semilla del `StratifiedKFold` interno (con un offset por fold externo).
- `model` (42): semilla de los modelos estocasticos (Bagging, AdaBoost, Gradient Boosting, Stacking).
- `search` (42): semilla de `RandomizedSearchCV`.

`k_outer = min(5, n_min)` adapta el numero de folds externos al soporte de la clase minoritaria. `k_inner = max(2, min(3, k_outer))`. Si el soporte de la clase minoritaria dentro de un fold externo es 1, el ciclo interno cae a `KFold` no estratificado (warning automatico).

## 11. Stacking robusto

`RobustStackingClassifier` (en `src/models.py`):

- OOF (`StratifiedKFold` interno) cuando `min_count >= 2`.
- In-sample cuando `min_count == 1`, aceptando un sesgo conocido documentado en `outputs/advertencias.txt`.
- Meta-clasificador: `LogisticRegression(class_weight="balanced")` con `final_C` configurable.
- `passthrough` opcional: concatena los 15 features originales a las meta-features antes del meta-clasificador (probado en el grid y en random).

## 12. Tests de significancia

- Wilcoxon firmado pareado sobre F1 macro por fold.
- McNemar con correccion de continuidad de Yates sobre los aciertos/fallos de cada par.
- Resultados en `outputs/tables/significance_tests_<experimento>.csv`.
- Heatmaps por objetivo y por test en `outputs/figures/analysis/significance_heatmap_<test>_<experimento>.png`.
- Solo se ejecutan si se uso `--keep-estimators` (los estimadores de cada fold externo se guardan en `outputs/estimator_cache/`).

## 13. Trabajo futuro

- Robustez de AdaBoost: en `GDS_R4` y `GDS_R5` (clases muy desbalanceadas con stump), `learning_rate` bajo y `n_estimators` alto producen clasificadores degenerados. Una alternativa es usar `class_weight="balanced"` en el stump base, o sustituir AdaBoost por `BalancedRandomForest`/`RUSBoost`.
- Mas estimadores en `random_all` para Gradient Boosting y Stacking.
- Profundizar en `passthrough` con un grid mas fino.
- Comparar con `HistGradientBoostingClassifier` (mas rapido y con soporte nativo para NaN).
