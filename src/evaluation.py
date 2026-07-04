from __future__ import annotations

from collections import Counter
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold

from .models import ModelSpec


def class_distribution(y: pd.Series) -> dict[int, int]:
    return {int(label): int(count) for label, count in y.value_counts().sort_index().items()}


def compute_outer_folds(y: pd.Series, max_outer_folds: int) -> tuple[int, int]:
    counts = y.value_counts()
    n_min = int(counts.min())
    k_outer = min(max_outer_folds, n_min)
    if k_outer < 2:
        raise ValueError(
            "No es posible aplicar validación cruzada estratificada: "
            "existe una clase con un solo ejemplo."
        )
    return n_min, k_outer


def unimplemented_result(
    target_name: str,
    model_spec: ModelSpec,
    distribution: dict[int, int],
    n_min: int,
    k_outer: int,
    experiment_name: str = "",
) -> dict[str, Any]:
    """Devuelve un resultado con todas las claves del implementado, valores None.

    Esto garantiza que `unimplemented_result` y `run_nested_cv` produzcan
    la misma estructura, lo que simplifica los reportes y el CSV final.
    """
    return {
        "experiment_name": experiment_name,
        "target": target_name,
        "model_key": model_spec.key,
        "model_name": model_spec.display_name,
        "implemented": False,
        "status": "No implementado",
        "message": model_spec.student_note,
        "class_distribution": distribution,
        "n_min": n_min,
        "k_outer": k_outer,
        "n_folds_valid": 0,
        "n_folds_failed": 0,
        "k_inner_requested": None,
        "search_type": None,
        "search_scoring": None,
        "search_n_iter": None,
        "search_params": None,
        "accuracy_mean": None,
        "accuracy_std": None,
        "balanced_accuracy_mean": None,
        "balanced_accuracy_std": None,
        "precision_macro_mean": None,
        "precision_macro_std": None,
        "recall_macro_mean": None,
        "recall_macro_std": None,
        "f1_macro_mean": None,
        "f1_macro_std": None,
        "stability": None,
        "stability_raw": None,
        "icn": None,
        "icn_raw": None,
        "best_params_mode": "No implementado",
        "best_params_counts": {},
        "warnings": [],
        "labels": sorted(distribution),
        "confusion_matrix": None,
        "classification_report": None,
        "best_scores_internal": None,
        "best_score_internal_mean": None,
        "best_score_internal_std": None,
        "fold_metrics": None,
        "fold_f1_external": None,
        "fold_delta_sesgo": None,
        "delta_sesgo": None,
    }


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    model_spec: ModelSpec,
    validation_config: dict[str, Any],
    search_config: dict[str, Any],
    random_state_config: dict[str, int],
    experiment_name: str = "",
    return_estimators: bool = False,
    estimator_cache_dir: Any = None,
) -> dict[str, Any] | tuple[dict[str, Any], list]:
    """Validación cruzada anidada unificada para grid o random.

    Retorna un dict con métricas, mejores hiperparámetros, ICN (calculado
    después por `assign_icn`), y opcionalmente guarda los estimadores
    de cada fold externo para tests de significancia sin reentrenar.
    """
    distribution = class_distribution(y)
    n_min, k_outer = compute_outer_folds(y, validation_config["max_outer_folds"])
    if not model_spec.implemented:
        return unimplemented_result(
            target_name=target_name,
            model_spec=model_spec,
            distribution=distribution,
            n_min=n_min,
            k_outer=k_outer,
            experiment_name=experiment_name,
        )

    if model_spec.pipeline is None:
        raise ValueError(
            f"El modelo {model_spec.key} está marcado como implementado, pero no tiene pipeline."
        )

    search_type = str(search_config.get("type", "grid")).lower()
    scoring_name = search_config.get(
        "scoring", validation_config.get("scoring", "f1_macro")
    )
    search_params = _search_space_from_config(model_spec.key, search_type, search_config)
    search_space_config = (
        search_config.get("params")
        if search_type == "grid"
        else search_config.get("distributions")
    )

    outer_cv = StratifiedKFold(
        n_splits=k_outer,
        shuffle=True,
        random_state=_random_state(random_state_config, "outer_cv"),
    )
    requested_inner = max(2, min(validation_config["max_inner_folds"], k_outer))
    scorer = _build_scorer(scoring_name)
    labels = sorted(distribution)

    fold_metrics: list[dict[str, float]] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    best_params_counter: Counter[str] = Counter()
    result_warnings: list[str] = []
    fold_best_scores: list[float] = []
    fold_best_estimators: list = [] if return_estimators else None
    fold_test_indices: list[list[int]] = [] if return_estimators else None
    n_folds_valid = 0
    n_folds_failed = 0

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner_cv, inner_warning = _build_inner_cv(
            y_train=y_train,
            requested_inner=requested_inner,
            random_state=_random_state(random_state_config, "inner_cv") + fold_idx,
            target_name=target_name,
        )
        if inner_warning and inner_warning not in result_warnings:
            result_warnings.append(inner_warning)

        search = _build_search_cv(
            estimator=clone(model_spec.pipeline),
            search_type=search_type,
            search_params=search_params,
            n_iter=search_config.get("n_iter"),
            scoring=scorer,
            cv=inner_cv,
            n_jobs=validation_config["n_jobs"],
            random_state=_random_state(random_state_config, "search"),
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                search.fit(X_train, y_train)
            except ValueError as exc:
                n_folds_failed += 1
                # Si todo el grid falla (e.g. AdaBoost degenerado),
                # registramos el error y saltamos este fold.
                msg = f"{target_name} fold {fold_idx}: {exc}"
                if msg not in result_warnings:
                    result_warnings.append(msg)
                fold_best_scores.append(float("nan"))
                if return_estimators and fold_best_estimators is not None:
                    fold_best_estimators.append(None)
                    fold_test_indices.append([int(i) for i in test_idx])
                fold_metrics.append({
                    "accuracy": float("nan"),
                    "balanced_accuracy": float("nan"),
                    "precision_macro": float("nan"),
                    "recall_macro": float("nan"),
                    "f1_macro": float("nan"),
                })
                continue

            for warning_item in caught_warnings:
                message = str(warning_item.message)
                if message not in result_warnings:
                    result_warnings.append(message)

            # Si todos los fits devolvieron NaN, saltamos el fold.
            if np.isnan(search.best_score_):
                n_folds_failed += 1
                msg = f"{target_name} fold {fold_idx}: todas las configs del search fallaron."
                if msg not in result_warnings:
                    result_warnings.append(msg)
                fold_best_scores.append(float("nan"))
                if return_estimators and fold_best_estimators is not None:
                    fold_best_estimators.append(None)
                    fold_test_indices.append([int(i) for i in test_idx])
                fold_metrics.append({
                    "accuracy": float("nan"),
                    "balanced_accuracy": float("nan"),
                    "precision_macro": float("nan"),
                    "recall_macro": float("nan"),
                    "f1_macro": float("nan"),
                })
                continue

        n_folds_valid += 1
        fold_best_scores.append(search.best_score_)
        if return_estimators and fold_best_estimators is not None:
            fold_best_estimators.append(search.best_estimator_)
            fold_test_indices.append([int(i) for i in test_idx])

        y_pred = search.predict(X_test)
        y_test_list = [int(value) for value in y_test.to_list()]
        y_pred_list = [int(value) for value in y_pred.tolist()]

        all_true.extend(y_test_list)
        all_pred.extend(y_pred_list)
        best_params_counter[_format_params(search.best_params_)] += 1

        fold_metrics.append(_compute_fold_metrics(y_test_list, y_pred_list))

    metrics_df = pd.DataFrame(fold_metrics)
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    report = classification_report(
        all_true,
        all_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "target": target_name,
        "experiment_name": experiment_name,
        "model_key": model_spec.key,
        "model_name": model_spec.display_name,
        "implemented": True,
        "status": "Implementado",
        "message": "",
        "class_distribution": distribution,
        "n_min": n_min,
        "k_outer": k_outer,
        "n_folds_valid": n_folds_valid,
        "n_folds_failed": n_folds_failed,
        "k_inner_requested": requested_inner,
        "search_type": search_type,
        "search_scoring": scoring_name,
        "search_n_iter": search_config.get("n_iter") if search_type == "random" else None,
        "search_params": search_space_config,
        "best_params_mode": _best_params_mode(best_params_counter),
        "best_params_counts": dict(best_params_counter),
        "warnings": result_warnings,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    if n_folds_failed > 0:
        msg = f"{target_name}: Corrida incompleta ({n_folds_failed}/{k_outer} folds fallidos). Métricas agregadas marcadas como NaN para evitar optimismo."
        if msg not in result_warnings:
            result_warnings.append(msg)

    for metric_name in metrics_df.columns:
        col = metrics_df[metric_name]
        valid = col.dropna()
        if n_folds_failed > 0:
            result[f"{metric_name}_mean"] = float("nan")
            result[f"{metric_name}_std"] = float("nan")
        else:
            result[f"{metric_name}_mean"] = float(valid.mean()) if not valid.empty else float("nan")
            result[f"{metric_name}_std"] = (
                float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
            )

    fold_f1_external = [m["f1_macro"] for m in fold_metrics]
    result["best_scores_internal"] = fold_best_scores
    valid_scores = [s for s in fold_best_scores if not np.isnan(s)]
    if n_folds_failed > 0:
        result["best_score_internal_mean"] = float("nan")
        result["best_score_internal_std"] = float("nan")
    else:
        result["best_score_internal_mean"] = (
            float(np.mean(valid_scores)) if valid_scores else float("nan")
        )
        result["best_score_internal_std"] = (
            float(np.std(valid_scores, ddof=1)) if len(valid_scores) > 1 else 0.0
        )
    result["fold_metrics"] = fold_metrics
    result["fold_f1_external"] = fold_f1_external
    result["fold_delta_sesgo"] = [
        (i - e) if (not np.isnan(i) and not np.isnan(e)) else float("nan")
        for i, e in zip(fold_best_scores, fold_f1_external)
    ]
    valid_deltas = [d for d in result["fold_delta_sesgo"] if not np.isnan(d)]
    if n_folds_failed > 0:
        result["delta_sesgo"] = float("nan")
    else:
        result["delta_sesgo"] = float(np.mean(valid_deltas)) if valid_deltas else float("nan")

    result["stability_raw"] = float(
        max(0.0, min(1.0, 1.0 - result["f1_macro_std"]))
    )
    result["icn"] = None
    result["icn_raw"] = None
    result["stability"] = result["stability_raw"]

    if return_estimators:
        if estimator_cache_dir is not None:
            from .significance import save_estimators
            save_estimators(
                estimator_cache_dir,
                target=target_name,
                model_key=model_spec.key,
                experiment_name=experiment_name,
                payload={
                    "estimators": fold_best_estimators,
                    "test_indices": fold_test_indices,
                },
            )
        return result, fold_best_estimators
    return result


ICN_WEIGHTS = {
    "f1_macro": 0.40,
    "balanced_accuracy": 0.25,
    "recall_macro": 0.20,
    "precision_macro": 0.10,
    "stability": 0.05,
}


def _compute_icn_raw(item: dict[str, Any]) -> float:
    """ICN sin normalizar

    Suma ponderada de F1 macro, balanced accuracy, recall macro,
    precision macro y estabilidad cruda. El valor esta en ~[0, 1]
    en la practica, comparable entre targets. Si alguna metrica es
    NaN, devuelve NaN para no propagar un promedio invalido.
    """
    parts = [
        ICN_WEIGHTS["f1_macro"] * item["f1_macro_mean"],
        ICN_WEIGHTS["balanced_accuracy"] * item["balanced_accuracy_mean"],
        ICN_WEIGHTS["recall_macro"] * item["recall_macro_mean"],
        ICN_WEIGHTS["precision_macro"] * item["precision_macro_mean"],
        ICN_WEIGHTS["stability"] * item["stability_raw"],
    ]
    if any(np.isnan(p) for p in parts):
        return float("nan")
    return float(sum(parts))


def assign_icn(results: list[dict[str, Any]]) -> None:
    """Asigna dos ICN a cada corrida:

    - ``icn_raw``: formula cruda (sin normalizar). Comparable
      entre targets, dominado por la magnitud absoluta de F1 y BA.
    - ``icn``: cada componente normalizada min-max entre las corridas
      del mismo target, luego ponderada. Adecuado para ordenar dentro
      de un mismo target; NO comparable entre targets.

    ``stability_raw`` es la version cruda (1 - std F1) y ``stability``
    es la version normalizada que participa en ``icn``.
    """
    implemented = [item for item in results if item["implemented"]]
    if not implemented:
        return

    for item in implemented:
        item["icn_raw"] = _compute_icn_raw(item)

    if len(implemented) == 1:
        item = implemented[0]
        item["icn"] = item["icn_raw"]
        item["icn_note"] = (
            "ICN directo porque solo hay una corrida implementada; "
            "coincide con icn_raw por falta de base de normalizacion."
        )
        return

    metric_keys = [
        "f1_macro_mean",
        "balanced_accuracy_mean",
        "recall_macro_mean",
        "precision_macro_mean",
        "f1_macro_std",
    ]
    normalized: dict[str, dict[tuple[str, str], float]] = {}
    for key in metric_keys:
        values = np.array([item[key] for item in implemented], dtype=float)
        # Si todos los valores son NaN, devolvemos un mapa vacío.
        if np.all(np.isnan(values)):
            normalized[key] = {(item.get("experiment_name", ""), item["model_key"]): float("nan") for item in implemented}
            continue
        min_value = float(np.nanmin(values))
        max_value = float(np.nanmax(values))
        denom = max_value - min_value + 1e-12
        normalized[key] = {}
        for item in implemented:
            run_key = (item.get("experiment_name", ""), item["model_key"])
            if np.isnan(item[key]):
                normalized[key][run_key] = float("nan")
                continue
            if key == "f1_macro_std":
                normalized[key][run_key] = float(
                    1.0 - (item[key] - min_value) / denom
                )
            else:
                normalized[key][run_key] = float(
                    (item[key] - min_value) / denom
                )

    for item in implemented:
        run_key = (item.get("experiment_name", ""), item["model_key"])
        item["stability"] = normalized["f1_macro_std"][run_key]
        components = [
            0.40 * normalized["f1_macro_mean"][run_key],
            0.25 * normalized["balanced_accuracy_mean"][run_key],
            0.20 * normalized["recall_macro_mean"][run_key],
            0.10 * normalized["precision_macro_mean"][run_key],
            0.05 * normalized["f1_macro_std"][run_key],
        ]
        if any(np.isnan(c) for c in components):
            item["icn"] = float("nan")
        else:
            item["icn"] = float(sum(components))
        item["icn_note"] = (
            "ICN normalizado min-max entre corridas del mismo target; útil "
            "para ordenar corridas dentro del target, no entre targets."
        )


def compute_delta_sesgo(results: list[dict[str, Any]]) -> None:
    """Delta sesgo = score interno (CV grid) - score externo (test fold).

    En `run_nested_cv` ya se calcula; esta función queda por compat
    hacia atrás y como safeguard si el caller no la pidió explícito.
    """
    for item in results:
        if not item.get("implemented", False):
            continue
        internal = item.get("best_score_internal_mean")
        external = item.get("f1_macro_mean")
        if internal is not None and external is not None:
            item["delta_sesgo"] = internal - external


def _search_space_from_config(
    model_key: str,
    search_type: str,
    search_config: dict[str, Any],
) -> dict[str, Any]:
    if search_type == "grid":
        params = search_config.get("params") or {}
        if not params:
            # Modelos sin hiperparámetros (dummy baseline). Devolvemos
            # un dict vacío para que GridSearchCV corra con un solo fit.
            return params
        return params

    if search_type == "random":
        distributions = search_config.get("distributions") or {}
        if not distributions:
            # Mismo caso: dummy u otro modelo sin hiperparámetros.
            return distributions
        return _materialize_distributions(distributions)

    raise ValueError(
        f"Tipo de búsqueda no soportado: {search_type}. Use 'grid' o 'random'."
    )


def _materialize_distributions(
    distributions_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        param_name: _materialize_distribution(param_name, distribution_config)
        for param_name, distribution_config in distributions_config.items()
    }


def _materialize_distribution(param_name: str, distribution_config: Any) -> Any:
    if not isinstance(distribution_config, dict):
        raise ValueError(
            f"La distribucion de {param_name} debe ser un diccionario. "
            "Use {'values': [...]} para valores discretos o {'dist': ...} para distribuciones."
        )

    if "values" in distribution_config:
        return distribution_config["values"]

    dist_name = distribution_config.get("dist")
    if dist_name == "randint":
        return randint(int(distribution_config["low"]), int(distribution_config["high"]))
    if dist_name == "uniform":
        low = float(distribution_config["low"])
        high = float(distribution_config["high"])
        return uniform(loc=low, scale=high - low)
    if dist_name == "loguniform":
        return loguniform(float(distribution_config["low"]), float(distribution_config["high"]))

    raise ValueError(
        f"Distribucion no soportada para {param_name}: {dist_name}. "
        "Use 'randint', 'uniform', 'loguniform' o 'values'."
    )


def _build_search_cv(
    estimator: Any,
    search_type: str,
    search_params: dict[str, list[Any]],
    n_iter: int | None,
    scoring: Any,
    cv: StratifiedKFold | KFold,
    n_jobs: int,
    random_state: int,
) -> GridSearchCV | RandomizedSearchCV:
    # `error_score=nan` permite que combinaciones inválidas
    # (e.g. AdaBoost con clases muy desbalanceadas) no rompan el flujo;
    # quedan con score NaN y se filtran al elegir `best_params_`.
    common_kwargs = {
        "estimator": estimator,
        "scoring": scoring,
        "cv": cv,
        "n_jobs": n_jobs,
        "refit": True,
        "error_score": np.nan,
    }

    if search_type == "grid":
        return GridSearchCV(param_grid=search_params, **common_kwargs)

    if search_type == "random":
        if not search_params:
            # Sin hiperparámetros: caemos a GridSearchCV con 1 sola config.
            return GridSearchCV(param_grid={}, **common_kwargs)
        if n_iter is None:
            raise ValueError("RandomizedSearchCV requiere configurar n_iter.")
        return RandomizedSearchCV(
            param_distributions=search_params,
            n_iter=int(n_iter),
            random_state=random_state,
            **common_kwargs,
        )

    raise ValueError(
        f"Tipo de búsqueda no soportado: {search_type}. Use 'grid' o 'random'."
    )


def _build_scorer(scoring_name: str) -> Any:
    scorers = {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }
    if scoring_name not in scorers:
        valid = ", ".join(sorted(scorers))
        raise ValueError(f"Scoring no soportado: {scoring_name}. Valores válidos: {valid}.")
    return scorers[scoring_name]


def _random_state(random_state_config: dict[str, int], key: str) -> int:
    if key in random_state_config:
        return int(random_state_config[key])
    return int(random_state_config.get("global", 42))


def _build_inner_cv(
    y_train: pd.Series,
    requested_inner: int,
    random_state: int,
    target_name: str,
) -> tuple[StratifiedKFold | KFold, str | None]:
    min_train_count = int(y_train.value_counts().min())
    if min_train_count >= requested_inner:
        return (
            StratifiedKFold(n_splits=requested_inner, shuffle=True, random_state=random_state),
            None,
        )
    if min_train_count >= 2:
        adjusted_inner = min(requested_inner, min_train_count)
        return (
            StratifiedKFold(n_splits=adjusted_inner, shuffle=True, random_state=random_state),
            (
                f"{target_name}: k_inner ajustado de {requested_inner} a {adjusted_inner} "
                f"por soporte mínimo {min_train_count} en entrenamiento externo."
            ),
        )

    return (
        KFold(n_splits=2, shuffle=True, random_state=random_state),
        (
            f"{target_name}: ciclo interno usa KFold no estratificado porque una clase tiene "
            "solo 1 ejemplo dentro de un entrenamiento externo."
        ),
    )


def _compute_fold_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _format_params(params: dict[str, Any]) -> str:
    if not params:
        return "default"
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def _best_params_mode(counter: Counter[str]) -> str:
    if not counter:
        return ""
    value, count = counter.most_common(1)[0]
    total = sum(counter.values())
    return f"{value} ({count}/{total})"
