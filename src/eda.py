"""Analisis exploratorio de datos (EDA) del dataset de deterioro cognitivo."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_loader import FEATURE_COLS, TARGETS

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.0)


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera las figuras y tablas estandar de EDA."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[eda] class distribution...")
    _plot_class_distribution(df, output_dir / "class_distribution.png")
    print("[eda] feature correlation...")
    _plot_feature_correlation(df, output_dir / "feature_correlation.png")
    print("[eda] target relationships...")
    _plot_target_relationships(df, output_dir / "target_relationships.png")
    print("[eda] eda summary...")
    _write_eda_summary(df, output_dir / "eda_summary.csv")
    print(f"[eda] completo. Figuras en {output_dir}")


def _plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Distribucion de clases por target con n_min y k_outer anotados."""
    n_targets = len(TARGETS)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = np.array(axes).flatten()

    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        counts = df[target].value_counts().sort_index()
        n_min = int(counts.min())
        k_outer = min(5, n_min)
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="colorblind")
        ax.set_title(f"{target} (n_min={n_min}, k_outer={k_outer})", fontweight="bold")
        ax.set_ylabel("Frecuencia")
        ax.set_xlabel("Clase")
        for patch_idx, value in enumerate(counts.values):
            ax.text(patch_idx, value, str(int(value)), ha="center", va="bottom", fontsize=8)

    for idx in range(len(TARGETS), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(
        "Distribucion de clases por objetivo (n_min controla k_outer)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_correlation(df: pd.DataFrame, output_path: Path) -> None:
    """Matriz de correlación phi (Pearson en binarias) entre los 15 features."""
    corr = df[FEATURE_COLS].astype(float).corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.7, "label": "Correlación"},
        ax=ax,
    )
    ax.set_title(
        "Correlacion entre los 15 atributos (Pearson = phi en binarias)",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_target_relationships(df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap que muestra como se derivan GDS_Rk a partir de GDS."""
    cross = pd.crosstab(df["GDS"], df["GDS_R1"], normalize="index")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cross, annot=True, fmt=".2f", cmap="YlOrRd",
        cbar_kws={"label": "Proporción dentro de GDS"}, ax=ax,
    )
    ax.set_title("GDS vs GDS_R1 (proporcion por clase de GDS)", fontweight="bold")
    ax.set_ylabel("GDS original")
    ax.set_xlabel("GDS_R1 reagrupado")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_eda_summary(df: pd.DataFrame, output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        counts = df[target].value_counts()
        n_min = int(counts.min())
        n_max = int(counts.max())
        n_classes = int(df[target].nunique())
        ratio = float(n_max) / float(n_min) if n_min > 0 else float("inf")
        rows.append({
            "target": target,
            "n": int(len(df)),
            "n_classes": n_classes,
            "n_min": n_min,
            "n_max": n_max,
            "imbalance_ratio": round(ratio, 1),
            "k_outer": min(5, n_min),
        })
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
