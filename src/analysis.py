"""Análisis avanzado: learning curves, feature importance, hiperparámetros,
significancia estadística, comparativa grid vs random, comparativa con Lab 3.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.tree import plot_tree

from .data_loader import FEATURE_COLS, TARGETS, load_sav_dataset
from .models import MODEL_DISPLAY_NAMES, MODEL_ORDER, build_model_registry
from .significance import load_estimators, run_significance_tests

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.0)

DEFAULT_LAB3_CSV = Path("Lab 3 Anterior/Lab03-ML-2026-01/outputs/tables/resumen_resultados.csv")


def run_advanced_analysis(
    results_by_target: dict[str, list[dict[str, Any]]],
    df: pd.DataFrame,
    output_dirs: dict[str, Path],
    estimator_cache_dir: Path | None = None,
    lab3_csv_path: Path | None = None,
) -> None:
    """Pipeline completo de analisis: figuras, tablas, tests."""
    figures_dir = output_dirs.get("analysis", output_dirs["figures"] / "analysis")
    tables_dir = output_dirs["tables"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[analysis] learning curves...")
    analyze_learning_curves(df, figures_dir)

    print("[analysis] feature importance...")
    analyze_feature_importance(df, figures_dir, estimator_cache_dir=estimator_cache_dir)

    print("[analysis] hyperparameter stability...")
    analyze_hyperparameter_stability(results_by_target, figures_dir)

    print("[analysis] zero-recall classes...")
    detect_zero_recall_classes(results_by_target, tables_dir, figures_dir)

    print("[analysis] low-support classes...")
    detect_low_support_classes(results_by_target, tables_dir)

    print("[analysis] per-fold stability...")
    analyze_per_fold_stability(results_by_target, figures_dir)

    print("[analysis] bootstrap CI...")
    analyze_bootstrap_ci(results_by_target, tables_dir, figures_dir)

    print("[analysis] decision tree (gradient boosting)...")
    visualize_gradient_boosting_trees(df, figures_dir, estimator_cache_dir=estimator_cache_dir)

    if estimator_cache_dir is not None:
        for exp_name in _experiment_names(results_by_target):
            print(f"[analysis] statistical tests ({exp_name})...")
            analyze_statistical_tests(
                results_by_target, df, tables_dir, figures_dir,
                estimator_cache_dir, experiment_name=exp_name,
            )
    else:
        print("[analysis] tests de significancia omitidos (sin cache de estimadores).")
        print("          Ejecute `python main.py --keep-estimators` para habilitarlos.")

    print("[analysis] grid vs random comparison...")
    analyze_grid_vs_random(results_by_target, tables_dir, figures_dir)

    print("[analysis] comparativa con Lab 3...")
    analyze_ensembles_vs_lab3(
        results_by_target,
        tables_dir,
        figures_dir,
        lab3_csv_path=lab3_csv_path or DEFAULT_LAB3_CSV,
    )

    print(f"[analysis] completo. Figuras en {figures_dir}")


def _experiment_names(results_by_target: dict[str, list[dict[str, Any]]]) -> list[str]:
    names: set[str] = set()
    for results in results_by_target.values():
        for item in results:
            name = item.get("experiment_name")
            if name:
                names.add(name)
    return sorted(names)


# --- Curvas de aprendizaje ---

def analyze_learning_curves(df: pd.DataFrame, output_dir: Path) -> None:
    registry = build_model_registry(random_state=42)
    n_targets = len(TARGETS)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = np.array(axes).flatten()

    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        X = df[FEATURE_COLS].astype(float)
        y = df[target].astype(int)

        counts = y.value_counts()
        n_min = int(counts.min())
        k_cv = max(2, min(5, n_min))

        for model_key in MODEL_ORDER:
            spec = registry[model_key]
            if not spec.implemented:
                continue
            try:
                train_sizes, _, test_scores = learning_curve(
                    clone(spec.pipeline),
                    X, y,
                    train_sizes=np.linspace(0.1, 1.0, 6),
                    cv=StratifiedKFold(n_splits=k_cv, shuffle=True, random_state=42),
                    scoring="f1_macro",
                    n_jobs=-1,
                    error_score=0.0,
                )
                mean_test = test_scores.mean(axis=1)
                std_test = test_scores.std(axis=1)
                ax.plot(
                    train_sizes, mean_test, marker="o",
                    label=spec.display_name, linewidth=1.5,
                )
                ax.fill_between(
                    train_sizes, mean_test - std_test, mean_test + std_test, alpha=0.1,
                )
            except Exception:
                continue

        ax.set_title(f"{target} (k={k_cv})", fontweight="bold")
        ax.set_xlabel("Tamano del entrenamiento")
        ax.set_ylabel("F1 macro")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")

    for idx in range(len(TARGETS), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(
        "Curvas de aprendizaje por target y ensamble",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Feature importance ---

def _extract_importance(estimator: Any, model_key: str) -> np.ndarray:
    """Devuelve la importancia por feature para un estimador del registry.

    Para Bagging promedia los `feature_importances_` de los árboles base
    reasignando al espacio completo (cada árbol vio un subconjunto por
    `max_features`). Stacking no expone importancias, devuelve ceros.
    """
    n_total = len(FEATURE_COLS)
    if model_key == "stacking":
        return np.zeros(n_total)
    if model_key == "bagging":
        clf = estimator.named_steps["clf"]
        aggregated = np.zeros(n_total)
        # Lista de features efectivamente usados por cada árbol,
        # en el orden en que aparecen en `tree.feature_importances_`.
        if hasattr(clf, "estimators_features_"):
            feature_subsets = clf.estimators_features_
        else:
            feature_subsets = [np.arange(n_total) for _ in clf.estimators_]
        for tree, subset in zip(clf.estimators_, feature_subsets, strict=True):
            for local_idx, global_idx in enumerate(subset[: len(tree.feature_importances_)]):
                aggregated[int(global_idx)] += tree.feature_importances_[local_idx]
        return aggregated / max(len(clf.estimators_), 1)
    if model_key in ("adaboost", "gradient_boosting"):
        return estimator.named_steps["clf"].feature_importances_
    return np.zeros(n_total)


def analyze_feature_importance(
    df: pd.DataFrame,
    output_dir: Path,
    estimator_cache_dir: Path | None = None,
) -> None:
    """Heatmap de importancia por feature y por ensamble."""
    importance_models = [
        ("bagging", "Bagging"),
        ("adaboost", "AdaBoost"),
        ("gradient_boosting", "Gradient Boosting"),
    ]
    registry = build_model_registry(random_state=42)

    for target in TARGETS:
        importance_data: dict[str, np.ndarray] = {}

        for model_key, display_name in importance_models:
            per_fold_imps: list[np.ndarray] = []
            if estimator_cache_dir is not None:
                for exp_name in _experiment_names_for_target(estimator_cache_dir, target, model_key):
                    payload = load_estimators(estimator_cache_dir, target, model_key, exp_name)
                    if payload is None:
                        continue
                    for est in payload["estimators"]:
                        try:
                            per_fold_imps.append(_extract_importance(est, model_key))
                        except Exception:
                            continue

            if not per_fold_imps:
                spec = registry[model_key]
                try:
                    model = clone(spec.pipeline)
                    model.fit(df[FEATURE_COLS].astype(float), df[target].astype(int))
                    per_fold_imps = [_extract_importance(model, model_key)]
                except Exception:
                    per_fold_imps = [np.zeros(len(FEATURE_COLS))]

            imp = np.mean(per_fold_imps, axis=0)
            imp = imp / (imp.sum() + 1e-12)
            importance_data[display_name] = imp

        imp_df = pd.DataFrame(importance_data, index=FEATURE_COLS)
        fig, ax = plt.subplots(
            figsize=(len(importance_models) * 2.5 + 2, len(FEATURE_COLS) * 0.45 + 2),
            constrained_layout=True,
        )
        sns.heatmap(
            imp_df, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5,
            cbar_kws={"shrink": 0.6, "label": "Importancia normalizada"},
            ax=ax,
        )
        ax.set_title(
            f"Importancia de atributos (promedio sobre folds) - {target}",
            fontweight="bold", fontsize=13,
        )
        ax.set_ylabel("Atributo")
        fig.savefig(output_dir / f"feature_importance_{target}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _experiment_names_for_target(
    cache_dir: Path, target: str, model_key: str
) -> list[str]:
    """Devuelve los nombres de experimentos que tienen cache para (target, modelo)."""
    safe_target = target.replace("/", "_")
    pattern = f"estimators_*_{safe_target}_{model_key}.joblib"
    found: list[str] = []
    for path in cache_dir.glob(pattern):
        # estimators_<exp>_<target>_<model>.joblib
        stem = path.stem  # estimators_<exp>_<target>_<model>
        parts = stem.split("_")
        if len(parts) >= 4:
            exp = "_".join(parts[1:-2])
            # exp puede contener "_" si el nombre lo trae
            found.append(exp)
    return found


# --- Estabilidad de hiperparámetros ---

def analyze_hyperparameter_stability(
    results_by_target: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    for target, results in results_by_target.items():
        data: list[dict[str, Any]] = []
        for item in results:
            if not item["implemented"]:
                continue
            data.append({
                "corrida": f"{item.get('experiment_name', '?')}/{item['model_name']}",
                "hiperparametros": item.get("best_params_mode", ""),
            })
        if not data:
            continue

        hp_df = pd.DataFrame(data)
        pivot = hp_df.groupby(["corrida", "hiperparametros"]).size().unstack(fill_value=0)
        if pivot.empty:
            continue

        fig, ax = plt.subplots(
            figsize=(max(8, len(pivot.columns) * 1.5), max(4, len(pivot.index) * 0.7)),
            constrained_layout=True,
        )
        sns.heatmap(
            pivot, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
            cbar_kws={"shrink": 0.6, "label": "Frecuencia (sobre folds externos)"},
            ax=ax,
        )
        ax.set_title(
            f"Estabilidad de hiperparámetros - {target}",
            fontweight="bold", fontsize=12,
        )
        fig.savefig(output_dir / f"hyperparam_stability_{target}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# --- Clases con recall cero / soporte bajo ---

def detect_zero_recall_classes(
    results_by_target: dict[str, list[dict[str, Any]]],
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for target, results in results_by_target.items():
        for item in results:
            if not item["implemented"] or not item.get("classification_report"):
                continue
            report = item["classification_report"]
            for label in item["labels"]:
                class_metrics = report.get(str(label), {})
                recall = class_metrics.get("recall", 0)
                if recall == 0.0:
                    rows.append({
                        "target": target,
                        "corrida": f"{item.get('experiment_name', '?')}/{item['model_name']}",
                        "clase": label,
                        "precision": class_metrics.get("precision", 0),
                        "recall": 0.0,
                        "f1_score": class_metrics.get("f1-score", 0),
                        "support": int(class_metrics.get("support", 0)),
                    })

    cols = ["target", "corrida", "clase", "precision", "recall", "f1_score", "support"]
    if not rows:
        pd.DataFrame(columns=cols).to_csv(tables_dir / "zero_recall_classes.csv", index=False, encoding="utf-8")
        return

    zero_df = pd.DataFrame(rows, columns=cols)
    zero_df.to_csv(tables_dir / "zero_recall_classes.csv", index=False, encoding="utf-8")

    pivot = zero_df.pivot_table(
        index=["target", "clase"], columns="corrida", values="recall", aggfunc="first"
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.2), max(4, len(pivot.index) * 0.5))
    )
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="Reds", linewidths=0.5,
        cbar_kws={"shrink": 0.6, "label": "Recall (0 = clase nunca predicha)"},
        ax=ax,
    )
    ax.set_title("Clases con recall = 0 por target y corrida", fontweight="bold", fontsize=12)
    ax.set_ylabel("Target - Clase")
    fig.tight_layout()
    fig.savefig(figures_dir / "zero_recall_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def detect_low_support_classes(
    results_by_target: dict[str, list[dict[str, Any]]],
    tables_dir: Path,
    min_support: int = 10,
) -> None:
    rows: list[dict[str, Any]] = []
    for target, results in results_by_target.items():
        if not results:
            continue
        distribution = results[0].get("class_distribution", {})
        for label, support in distribution.items():
            if support >= min_support:
                continue
            for item in results:
                recall = precision = f1 = None
                zero_recall = False
                if item["implemented"] and item.get("classification_report"):
                    cr = item["classification_report"]
                    class_m = cr.get(str(int(label)), {})
                    recall = class_m.get("recall")
                    precision = class_m.get("precision")
                    f1 = class_m.get("f1-score")
                    zero_recall = (recall == 0.0)
                rows.append({
                    "target": target,
                    "clase": int(label),
                    "support": int(support),
                    "corrida": f"{item.get('experiment_name', '?')}/{item['model_name']}",
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "zero_recall": zero_recall,
                })

    cols = ["target", "clase", "support", "corrida", "precision", "recall", "f1_score", "zero_recall"]
    pd.DataFrame(rows, columns=cols).to_csv(tables_dir / "low_support_classes.csv", index=False, encoding="utf-8")


# --- Distribución por fold ---

def analyze_per_fold_stability(
    results_by_target: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    n_targets = len(TARGETS)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = np.array(axes).flatten()

    for idx, (target, results) in enumerate(results_by_target.items()):
        ax = axes[idx]
        data: list[dict[str, Any]] = []
        for item in results:
            if not item["implemented"] or not item.get("fold_metrics"):
                continue
            for fold_idx, fm in enumerate(item["fold_metrics"]):
                data.append({
                    "corrida": f"{item.get('experiment_name', '?')}/{item['model_name']}",
                    "fold": fold_idx + 1,
                    "f1_macro": fm["f1_macro"],
                })
        if not data:
            ax.set_title(target)
            ax.text(0.5, 0.5, "Sin datos por fold", ha="center", va="center", transform=ax.transAxes)
            continue
        df_plot = pd.DataFrame(data)
        sns.boxplot(
            data=df_plot, x="corrida", y="f1_macro", hue="corrida",
            ax=ax, palette="colorblind", legend=False,
        )
        ax.set_title(target, fontweight="bold")
        ax.set_ylabel("F1 macro")
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Distribucion de F1 macro por fold externo", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "per_fold_f1_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Bootstrap CI ---

def bootstrap_ci(
    metric_values: list[float], n_bootstrap: int = 1000, ci: float = 0.95
) -> tuple[float, float]:
    arr = np.asarray(metric_values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed=42)
    boot_means = np.empty(n_bootstrap)
    n = len(arr)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = 1.0 - ci
    return (
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def analyze_bootstrap_ci(
    results_by_target: dict[str, list[dict[str, Any]]],
    tables_dir: Path,
    figures_dir: Path,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> None:
    metrics = [
        ("f1_macro", "F1 macro"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("recall_macro", "Recall macro"),
    ]
    rows: list[dict[str, Any]] = []
    forest_data: list[dict[str, Any]] = []
    for target, results in results_by_target.items():
        for item in results:
            if not item["implemented"] or not item.get("fold_metrics"):
                continue
            for metric_key, metric_label in metrics:
                values = [m.get(metric_key) for m in item["fold_metrics"]]
                values = [v for v in values if v is not None]
                if not values:
                    continue
                ci_low, ci_high = bootstrap_ci(values, n_bootstrap=n_bootstrap, ci=ci)
                rows.append({
                    "target": target,
                    "corrida": f"{item.get('experiment_name', '?')}/{item['model_name']}",
                    "metrica": metric_label,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "n_folds": len(values),
                })
                if metric_key == "f1_macro":
                    forest_data.append({
                        "target": target,
                        "corrida": f"{item.get('experiment_name', '?')}/{item['model_name']}",
                        "mean": float(np.mean(values)),
                        "ci_lower": ci_low,
                        "ci_upper": ci_high,
                    })

    ci_pct = int(ci * 100)
    pd.DataFrame(rows).to_csv(tables_dir / f"bootstrap_ci_{ci_pct}.csv", index=False, encoding="utf-8")
    if not forest_data:
        return

    forest_df = pd.DataFrame(forest_data)
    n_targets = len(TARGETS)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 5))
    axes = np.array(axes).flatten()

    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        sub = forest_df[forest_df["target"] == target].copy()
        if sub.empty:
            ax.set_title(target)
            continue
        order = sub.sort_values("mean", ascending=True)
        y_pos = np.arange(len(order))
        for y_i, (_, row) in enumerate(order.iterrows()):
            ax.errorbar(
                row["mean"], y_i,
                xerr=[[row["mean"] - row["ci_lower"]], [row["ci_upper"] - row["mean"]]],
                fmt="o", capsize=4, markersize=6, linewidth=1.5,
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(order["corrida"].tolist(), fontsize=7)
        ax.set_title(target, fontweight="bold")
        ax.set_xlabel("F1 macro")
        ax.grid(True, axis="x", alpha=0.3)

    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(
        f"F1 macro con IC bootstrap al {ci_pct}% por corrida y objetivo",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / f"bootstrap_ci_forest_{ci_pct}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Árbol de Gradient Boosting visualizado ---

def visualize_gradient_boosting_trees(
    df: pd.DataFrame,
    output_dir: Path,
    estimator_cache_dir: Path | None = None,
) -> None:
    """Visualiza el primer arbol del GradientBoostingClassifier por target.

    Gradient Boosting entrena muchos arboles en secuencia; mostramos el
    primero (el que se ajusta al residuo inicial) para inspeccionar la
    estructura del problema.
    """
    registry = build_model_registry(random_state=42)
    spec = registry["gradient_boosting"]
    for target in TARGETS:
        X = df[FEATURE_COLS].astype(float)
        y = df[target].astype(int)
        unique_classes = sorted(y.unique())

        tree = None
        if estimator_cache_dir is not None:
            for exp_name in _experiment_names_for_target(estimator_cache_dir, target, "gradient_boosting"):
                payload = load_estimators(estimator_cache_dir, target, "gradient_boosting", exp_name)
                if payload is None or not payload["estimators"]:
                    continue
                gb = payload["estimators"][0].named_steps["clf"]
                if hasattr(gb, "estimators_") and gb.estimators_.shape[0] > 0:
                    tree = gb.estimators_[0, 0]  # primer arbol, clase 0
                    suffix = f"primer arbol (mejor fold externo, exp={exp_name})"
                    break

        if tree is None:
            gb = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, random_state=42,
            )
            gb.fit(X, y)
            tree = gb.estimators_[0, 0]
            suffix = "primer arbol (fallback global, n_estimators=50)"

        fig, ax = plt.subplots(figsize=(20, 12))
        plot_tree(
            tree, feature_names=FEATURE_COLS,
            class_names=[str(c) for c in unique_classes],
            filled=True, rounded=True, fontsize=9, ax=ax,
        )
        ax.set_title(
            f"Gradient Boosting - primer arbol - {target} ({suffix})",
            fontweight="bold", fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"gb_tree_{target}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# --- Tests de significancia ---

def analyze_statistical_tests(
    results_by_target: dict[str, list[dict[str, Any]]],
    df: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    estimator_cache_dir: Path,
    alpha: float = 0.05,
    experiment_name: str = "grid_all",
) -> None:
    target_names = [t for t in results_by_target if t in TARGETS]
    rows_df = run_significance_tests(
        df=df,
        results_by_target=results_by_target,
        cache_dir=estimator_cache_dir,
        targets=target_names,
        model_order=MODEL_ORDER,
        model_display_names=MODEL_DISPLAY_NAMES,
        alpha=alpha,
        experiment_name=experiment_name,
    )
    if not rows_df.empty:
        rows_df.to_csv(
            tables_dir / f"significance_tests_{experiment_name}.csv",
            index=False, encoding="utf-8",
        )

    if rows_df.empty:
        return

    heatmap_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for target in target_names:
        sub = rows_df[rows_df["target"] == target]
        if sub.empty:
            continue
        models = sorted(set(sub["modelo_A"]).union(sub["modelo_B"]))
        p_w = pd.DataFrame(np.nan, index=models, columns=models)
        p_m = pd.DataFrame(np.nan, index=models, columns=models)
        for _, row in sub.iterrows():
            p_w.loc[row["modelo_A"], row["modelo_B"]] = row["p_wilcoxon"]
            p_w.loc[row["modelo_B"], row["modelo_A"]] = row["p_wilcoxon"]
            p_m.loc[row["modelo_A"], row["modelo_B"]] = row["p_mcnemar_yates"]
            p_m.loc[row["modelo_B"], row["modelo_A"]] = row["p_mcnemar_yates"]
        for name in models:
            p_w.loc[name, name] = np.nan
            p_m.loc[name, name] = np.nan
        heatmap_data[target] = (p_w, p_m)

    _plot_significance_heatmaps(heatmap_data, figures_dir, alpha, experiment_name)


def _plot_significance_heatmaps(
    heatmap_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    figures_dir: Path,
    alpha: float,
    experiment_name: str,
) -> None:
    if not heatmap_data:
        return
    n_targets = len(heatmap_data)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    for kind, label in (("wilcoxon", "Wilcoxon"), ("mcnemar", "McNemar-Yates")):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
        axes = np.array(axes).flatten()

        for idx, (target, (p_w, p_m)) in enumerate(heatmap_data.items()):
            ax = axes[idx]
            p = p_w if kind == "wilcoxon" else p_m
            if p.empty:
                ax.set_title(target)
                continue
            annot = p.copy().astype(object)
            for r in range(p.shape[0]):
                for c in range(p.shape[1]):
                    val = p.iloc[r, c]
                    if np.isnan(val):
                        annot.iloc[r, c] = ""
                    elif r == c:
                        annot.iloc[r, c] = "---"
                    else:
                        annot.iloc[r, c] = f"{val:.3f}{'*' if val < alpha else ''}"

            sns.heatmap(
                p.astype(float), annot=annot, fmt="", cmap="RdYlGn_r",
                vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={"shrink": 0.6, "label": f"p-value ({label})"},
                ax=ax,
            )
            ax.set_title(f"{target} ({experiment_name})", fontweight="bold", fontsize=10)
            ax.tick_params(axis="x", rotation=35, labelsize=7)
            ax.tick_params(axis="y", rotation=0, labelsize=7)

        for idx in range(n_targets, len(axes)):
            axes[idx].set_visible(False)
        fig.suptitle(
            f"{label}: p-values por par de ensambles ({experiment_name})",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"significance_heatmap_{kind}_{experiment_name}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)


# --- Comparativa grid_all vs random_all ---

def analyze_grid_vs_random(
    results_by_target: dict[str, list[dict[str, Any]]],
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    """Heatmap de F1 macro por (objetivo, modelo) con grid vs random.

    Genera:
    - `grid_vs_random_heatmap.png`: heatmap lado a lado.
    - `grid_vs_random_diff_heatmap.png`: diferencia (random - grid).
    - `grid_vs_random_summary.csv`: tabla con la mejor corrida por
      (objetivo, modelo).
    """
    rows: list[dict[str, Any]] = []
    for target, results in results_by_target.items():
        for item in results:
            if not item["implemented"] or item.get("f1_macro_mean") is None:
                continue
            rows.append({
                "target": target,
                "experiment": item.get("experiment_name", "?"),
                "model_key": item["model_key"],
                "model": MODEL_DISPLAY_NAMES.get(item["model_key"], item["model_key"]),
                "f1_macro": item["f1_macro_mean"],
                "f1_macro_std": item.get("f1_macro_std", 0.0),
                "icn": item.get("icn"),
                "search_type": item.get("search_type"),
                "search_n_iter": item.get("search_n_iter"),
            })
    if not rows:
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index=["target", "experiment"], columns="model", values="f1_macro"
    )
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return

    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.6), max(5, len(pivot.index) * 0.6))
    )
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "F1 macro"},
        ax=ax,
    )
    ax.set_title("F1 macro: grid_all vs random_all", fontweight="bold", fontsize=12)
    ax.set_ylabel("Objetivo / experimento")
    ax.set_xlabel("Modelo")
    fig.tight_layout()
    fig.savefig(figures_dir / "grid_vs_random_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Diferencia random - grid_all
    if "grid_all" in pivot.index.get_level_values(1) and "random_all" in pivot.index.get_level_values(1):
        grid = pivot.xs("grid_all", level="experiment")
        rand = pivot.xs("random_all", level="experiment")
        diff = (rand - grid).dropna(how="all", axis=1).dropna(how="all", axis=0)
        if not diff.empty:
            fig, ax = plt.subplots(
                figsize=(max(8, len(diff.columns) * 1.6), max(4, len(diff.index) * 0.6))
            )
            sns.heatmap(
                diff, annot=True, fmt="+.3f", cmap="RdBu_r", center=0,
                linewidths=0.5,
                cbar_kws={"label": "F1 (random_all) - F1 (grid_all)"},
                ax=ax,
            )
            ax.set_title(
                "Diferencia de F1 macro: random_all - grid_all (positivo = random gana)",
                fontweight="bold", fontsize=12,
            )
            ax.set_ylabel("Objetivo")
            ax.set_xlabel("Modelo")
            fig.tight_layout()
            fig.savefig(
                figures_dir / "grid_vs_random_diff_heatmap.png",
                dpi=300, bbox_inches="tight",
            )
            plt.close(fig)

    # Resumen tabular
    summary = df.pivot_table(
        index=["target", "model"], columns="experiment", values="f1_macro"
    ).reset_index()
    summary.to_csv(tables_dir / "grid_vs_random_summary.csv", index=False, encoding="utf-8")


# --- Comparativa con Lab 3 ---

def analyze_ensembles_vs_lab3(
    results_by_target: dict[str, list[dict[str, Any]]],
    tables_dir: Path,
    figures_dir: Path,
    lab3_csv_path: Path = DEFAULT_LAB3_CSV,
) -> None:
    """Lee el CSV de Lab 3 y compara F1 macro con los ensambles de Lab 4.

    Para cada target reporta el mejor F1 macro de Lab 3, el mejor F1
    macro entre los ensambles de Lab 4, y la diferencia absoluta y
    relativa.
    """
    if not lab3_csv_path.exists():
        print(f"[analysis] CSV de Lab 3 no encontrado en {lab3_csv_path}; comparativa omitida.")
        return

    lab3_df = pd.read_csv(lab3_csv_path)
    if "f1_macro_mean" not in lab3_df.columns or "target" not in lab3_df.columns:
        print("[analysis] CSV de Lab 3 no tiene f1_macro_mean / target; comparativa omitida.")
        return

    # Mejor de Lab 3 por target
    best_lab3 = (
        lab3_df.groupby("target")["f1_macro_mean"].max().rename("best_lab3_f1")
    )

    # Mejor ensamble por target (todos los modelos y experimentos)
    current_rows: list[dict[str, Any]] = []
    for target, results in results_by_target.items():
        for item in results:
            if not item["implemented"] or item.get("f1_macro_mean") is None:
                continue
            current_rows.append({
                "target": target,
                "corrida": f"{item.get('experiment_name', '?')}/{item['model_name']}",
                "f1_macro": item["f1_macro_mean"],
                "f1_macro_std": item.get("f1_macro_std", 0.0),
                "icn": item.get("icn"),
            })
    if not current_rows:
        return

    current_df = pd.DataFrame(current_rows)
    best_current_idx = current_df.groupby("target")["f1_macro"].idxmax()
    best_current = current_df.loc[best_current_idx].set_index("target")[
        ["corrida", "f1_macro", "f1_macro_std"]
    ].rename(columns={
        "f1_macro": "best_lab4_f1",
        "corrida": "best_lab4_corrida",
        "f1_macro_std": "best_lab4_std",
    })

    comparison = best_lab3.to_frame().join(best_current, how="outer")
    comparison["delta_abs"] = comparison["best_lab4_f1"] - comparison["best_lab3_f1"]
    comparison["delta_rel_pct"] = (
        100.0 * comparison["delta_abs"] / comparison["best_lab3_f1"]
    )
    comparison = comparison.reset_index()
    comparison.to_csv(
        tables_dir / "lab3_vs_lab4_comparison.csv", index=False, encoding="utf-8"
    )

    # Figura: F1 macro por target, comparativa entre Lab 3 y Lab 4.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(comparison))
    width = 0.38
    ax.bar(x - width / 2, comparison["best_lab3_f1"], width,
           label="Lab 3 (mejor)", color="#1f77b4")
    ax.bar(
        x + width / 2, comparison["best_lab4_f1"], width,
        yerr=comparison["best_lab4_std"], capsize=4,
        label="Lab 4 (mejor ensamble)", color="#ff7f0e",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["target"], rotation=0)
    ax.set_ylabel("F1 macro")
    ax.set_ylim(0, 1.0)
    ax.set_title("Comparativa Lab 3 vs Lab 4 (F1 macro por objetivo)", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, row in comparison.iterrows():
        delta = row["delta_abs"]
        ax.text(
            i, max(row["best_lab3_f1"], row["best_lab4_f1"]) + 0.02,
            f"Δ={delta:+.3f}", ha="center", fontsize=8, color="black",
        )
    fig.tight_layout()
    fig.savefig(figures_dir / "lab3_vs_lab4_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
