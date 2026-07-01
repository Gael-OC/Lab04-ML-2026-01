"""Tests de significancia estadística y cache de estimadores.

Wilcoxon firmado pareado: test no paramétrico sobre F1 macro fold a fold.
Válido con muestras pequeñas (k=2 o k=5).

McNemar con corrección de continuidad de Yates: discrepancias de acierto
entre dos modelos en los mismos ejemplos. Yates se aplica porque
b + c < 25 es común con datasets pequeños.

El cache de estimadores se llena cuando `run_nested_cv` corre con
`return_estimators=True` y un `estimator_cache_dir`. Esto desacopla
el costo de los tests de significancia del costo de los experimentos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2, wilcoxon

from .data_loader import FEATURE_COLS


def estimator_cache_path(
    cache_dir: Path, target: str, model_key: str, experiment_name: str = ""
) -> Path:
    """Ruta al archivo joblib con los estimadores de un (target, modelo, experimento)."""
    safe_target = target.replace("/", "_")
    safe_exp = experiment_name.replace("/", "_") if experiment_name else "default"
    return cache_dir / f"estimators_{safe_exp}_{safe_target}_{model_key}.joblib"


def save_estimators(
    cache_dir: Path,
    target: str,
    model_key: str,
    experiment_name: str,
    payload: dict[str, Any],
) -> None:
    """Guarda estimadores e índices de test para un (target, modelo, experimento)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = estimator_cache_path(cache_dir, target, model_key, experiment_name)
    joblib.dump(payload, path, compress=3)


def load_estimators(
    cache_dir: Path, target: str, model_key: str, experiment_name: str = ""
) -> dict[str, Any] | None:
    """Carga estimadores e índices de test. None si no existe cache."""
    path = estimator_cache_path(cache_dir, target, model_key, experiment_name)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _reconstruct_fold_predictions(
    estimators: list, X: pd.DataFrame, test_indices_by_fold: list[list[int]]
) -> list[np.ndarray]:
    """Re-genera predicciones por fold externo usando los estimadores entrenados.

    Es equivalente a llamar `est.predict(X_test)` para cada fold, pero
    centralizado para que los tests de significancia no reentrenen nada.
    """
    preds_per_fold = []
    for est, test_idx in zip(estimators, test_indices_by_fold):
        preds_per_fold.append(est.predict(X.iloc[test_idx]))
    return preds_per_fold


def _wilcoxon_pvalue(f1_a: list[float], f1_b: list[float]) -> float:
    """Wilcoxon signed-rank pareado. Devuelve np.nan si no se puede calcular."""
    if len(f1_a) < 2 or len(f1_b) < 2:
        return float("nan")
    diffs = np.array(f1_a) - np.array(f1_b)
    if np.all(diffs == 0):
        return 1.0
    try:
        _, p = wilcoxon(diffs, zero_method="zsplit", alternative="two-sided")
        return float(p)
    except ValueError:
        return float("nan")


def _mcnemar_yates_pvalue(b: int, c: int) -> float:
    """McNemar con corrección de continuidad de Yates."""
    if (b + c) == 0:
        return 1.0
    num = (abs(b - c) - 1) ** 2
    stat = num / (b + c)
    return float(1.0 - chi2.cdf(stat, df=1))


def _perfold_mcnemar_counts(
    preds_a_per_fold: list[np.ndarray],
    preds_b_per_fold: list[np.ndarray],
    y_true_per_fold: list[np.ndarray],
) -> tuple[int, int]:
    """Cuenta discrepancias agregadas a lo largo de los folds externos."""
    b_total = 0
    c_total = 0
    for preds_a, preds_b, y_true in zip(
        preds_a_per_fold, preds_b_per_fold, y_true_per_fold
    ):
        correct_a = preds_a == y_true
        correct_b = preds_b == y_true
        b_total += int(np.sum(correct_a & ~correct_b))
        c_total += int(np.sum(~correct_a & correct_b))
    return b_total, c_total


def run_significance_tests(
    df: pd.DataFrame,
    results_by_target: dict[str, list[dict[str, Any]]],
    cache_dir: Path,
    targets: list[str],
    model_order: list[str],
    model_display_names: dict[str, str],
    alpha: float = 0.05,
    experiment_name: str = "",
) -> pd.DataFrame:
    """Ejecuta Wilcoxon y McNemar (con Yates) sobre los folds externos cacheados.

    Parametros
    ----------
    df : DataFrame con las 15 features y las columnas objetivo.
    results_by_target : resultados previos (para reutilizar f1 por fold).
    cache_dir : directorio donde `run_nested_cv` guardo los estimadores.
    targets : lista de targets a evaluar.
    model_order : orden canonico de los modelos.
    model_display_names : mapeo model_key -> nombre legible.
    alpha : nivel de significancia.
    experiment_name : nombre del experimento (grid_all, random_all, etc.).

    Retorna
    -------
    DataFrame con una fila por par de modelos por target, con p-values
    Wilcoxon y McNemar-Yates y la diferencia media de F1 macro.
    """
    rows: list[dict[str, Any]] = []

    for target in targets:
        if target not in results_by_target:
            continue
        y = df[target].astype(int)
        X = df[FEATURE_COLS].astype(float)

        # f1 por fold y por modelo, en el orden de `model_order`
        f1_by_model: dict[str, list[float]] = {}
        for item in results_by_target[target]:
            if (
                item["implemented"]
                and item.get("fold_metrics")
                and item.get("experiment_name", "") == experiment_name
            ):
                f1_by_model[item["model_key"]] = [
                    m["f1_macro"] for m in item["fold_metrics"]
                ]

        preds_by_model: dict[str, list[np.ndarray]] = {}
        y_true_per_fold_by_model: dict[str, list[np.ndarray]] = {}
        for model_key in model_order:
            payload = load_estimators(cache_dir, target, model_key, experiment_name)
            if payload is None:
                continue
            estimators = payload["estimators"]
            test_indices = payload["test_indices"]
            preds_by_model[model_key] = _reconstruct_fold_predictions(
                estimators, X, test_indices
            )
            y_true_per_fold_by_model[model_key] = [
                y.iloc[test_idx].to_numpy() for test_idx in test_indices
            ]

        implemented_keys = [k for k in model_order if k in f1_by_model]

        for i, key_a in enumerate(implemented_keys):
            for j, key_b in enumerate(implemented_keys):
                if i >= j:
                    continue
                f1_a = f1_by_model[key_a]
                f1_b = f1_by_model[key_b]
                p_w = _wilcoxon_pvalue(f1_a, f1_b)

                preds_a = preds_by_model.get(key_a)
                preds_b = preds_by_model.get(key_b)
                p_m = float("nan")
                if (
                    preds_a is not None
                    and preds_b is not None
                    and len(preds_a) == len(preds_b)
                ):
                    y_true_per_fold = y_true_per_fold_by_model[key_a]
                    b, c = _perfold_mcnemar_counts(preds_a, preds_b, y_true_per_fold)
                    p_m = _mcnemar_yates_pvalue(b, c)

                rows.append(
                    {
                        "target": target,
                        "experiment_name": experiment_name,
                        "modelo_A": model_display_names[key_a],
                        "modelo_B": model_display_names[key_b],
                        "p_wilcoxon": p_w,
                        "p_mcnemar_yates": p_m,
                        "significativo_wilcoxon": (
                            p_w < alpha if not np.isnan(p_w) else False
                        ),
                        "significativo_mcnemar_yates": (
                            p_m < alpha if not np.isnan(p_m) else False
                        ),
                        "diferencia_media_f1": float(np.mean(f1_a) - np.mean(f1_b)),
                    }
                )

    return pd.DataFrame(rows)
