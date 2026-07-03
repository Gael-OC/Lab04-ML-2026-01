"""Visualizaciones principales del laboratorio: heatmaps y figuras comparativas."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .models import MODEL_DISPLAY_NAMES, MODEL_ORDER

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.0)


def generate_all_plots(
    results_by_target: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    """Heatmap F1 macro por (objetivo, modelo, experimento)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[plots] F1 macro heatmap...")
    _plot_f1_macro_heatmap(results_by_target, output_dir / "f1_macro_heatmap.png")
    print("[plots] completo.")


def _plot_f1_macro_heatmap(
    results_by_target: dict[str, list[dict[str, Any]]], output_path: Path
) -> None:
    """Heatmap de F1 macro: filas = (target, experimento), columnas = modelo.

    Se grafica el F1 macro de cada corrida. Los `experiment_name` se
    preservan en el indice, lo que permite comparar grid vs random.
    """
    rows: list[dict[str, Any]] = []
    for target, results in results_by_target.items():
        for item in results:
            if not item["implemented"] or item.get("f1_macro_mean") is None:
                continue
            rows.append({
                "target": target,
                "experiment": item.get("experiment_name", "?"),
                "model": MODEL_DISPLAY_NAMES.get(item["model_key"], item["model_key"]),
                "f1_macro": item["f1_macro_mean"],
            })
    if not rows:
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index=["target", "experiment"], columns="model", values="f1_macro"
    )

    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.6), max(5, len(pivot.index) * 0.6))
    )
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGnBu",
        vmin=0, vmax=1, linewidths=0.5,
        cbar_kws={"label": "F1 macro (promedio sobre folds externos)"},
        ax=ax,
    )
    ax.set_title("F1 macro por objetivo, experimento y modelo", fontweight="bold", fontsize=12)
    ax.set_ylabel("Objetivo / experimento")
    ax.set_xlabel("Modelo")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
