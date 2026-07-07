from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import textwrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# Columnas del CSV resumen.
SUMMARY_COLUMNS: list[str] = [
    # Identificación
    "experiment_name",
    "target",
    "model_name",
    "status",
    "n_min",
    "k_outer",
    "n_folds_valid",
    "n_folds_failed",
    # Métricas
    "f1_macro_mean",
    "f1_macro_std",
    "balanced_accuracy_mean",
    "recall_macro_mean",
    "precision_macro_mean",
    # Estabilidad e ICN
    "stability_raw",
    "stability",
    "icn_raw",
    "icn",
    # Sesgo
    "best_score_internal_mean",
    "delta_sesgo",
    # Búsqueda de hiperparámetros
    "search_type",
    "search_scoring",
    "search_n_iter",
    "best_params_mode",
    # Miscelánea
    "message",
]


def write_summary_csv(
    results_by_target: dict[str, list[dict[str, Any]]], output_path: Path
) -> None:
    rows = [item for results in results_by_target.values() for item in results]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: _csv_value(row.get(column)) for column in SUMMARY_COLUMNS}
            )


def write_json_results(
    results_by_target: dict[str, list[dict[str, Any]]], output_path: Path
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results_by_target, handle, ensure_ascii=False, indent=2)


def write_auxiliary_tables(
    results_by_target: dict[str, list[dict[str, Any]]], output_dirs: dict[str, Path]
) -> None:
    distributions_path = output_dirs["tables"] / "distribucion_clases.csv"
    with distributions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "class", "support"])
        for target, results in results_by_target.items():
            distribution = results[0]["class_distribution"]
            for label, support in distribution.items():
                writer.writerow([target, label, support])

    for target, results in results_by_target.items():
        for item in results:
            if not item["implemented"]:
                continue
            _write_confusion_matrix(target, item, output_dirs["confusion_matrices"])
            _write_class_report(target, item, output_dirs["per_class"])


def write_warnings(
    results_by_target: dict[str, list[dict[str, Any]]], output_path: Path
) -> None:
    lines: list[str] = []
    for target, results in results_by_target.items():
        for item in results:
            for warning in item.get("warnings", []):
                experiment_name = item.get("experiment_name", "experiment")
                lines.append(
                    f"[{experiment_name} | {target} | {item['model_name']}] {warning}"
                )
    if not lines:
        lines.append("No se registraron advertencias.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_tables(
    results_by_target: dict[str, list[dict[str, Any]]], output_path: Path
) -> None:
    lines: list[str] = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[spanish]{babel}",
        r"\usepackage{booktabs}",
        r"\usepackage{geometry}",
        r"\geometry{margin=1.5cm, landscape}",
        r"\begin{document}",
        r"\section*{Laboratorio 04: clasificadores de ensamble}",
        (
            "Bagging, AdaBoost, Stacking y Gradient Boosting evaluados con "
            "validacion cruzada anidada. Se reportan F1 macro, balanced "
            "accuracy, ICN crudo y normalizado, estabilidad y "
            "$\\Delta$sesgo, además de las columnas de búsqueda "
            r"(experimento, search\_type, search\_scoring, search\_n\_iter, "
            r"best\_params\_mode)."
        ),
        "",
    ]

    for target, results in results_by_target.items():
        distribution = _format_distribution(results[0]["class_distribution"])
        lines.extend(
            [
                rf"\subsection*{{Experimento {latex_escape(target)}}}",
                rf"\noindent\textbf{{Distribucion de clases:}} {latex_escape(distribution)}. "
                rf"\textbf{{n\_min:}} {results[0]['n_min']}. "
                rf"\textbf{{k externo:}} {results[0]['k_outer']}.",
                r"\begin{table}[h]",
                r"\centering",
                rf"\caption{{Resultados para {latex_escape(target)}}}",
                r"\scriptsize",
                r"\begin{tabular}{p{3.2cm}p{1.6cm}p{1.2cm}p{1.2cm}p{1.2cm}p{0.9cm}p{0.9cm}p{0.9cm}p{1.0cm}p{1.2cm}p{3.5cm}}",
                r"\toprule",
                (
                    r"Experimento / modelo & F1 macro & BalAcc & Recall & "
                    r"Precisi\'on & ICN* & ICN & $\Delta$sesgo & Busq. & n\_iter & "
                    r"Hiperpar\'ametros / estado \\"
                ),
                r"\midrule",
            ]
        )
        for item in results:
            lines.append(_latex_row(item))
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        )

    lines.extend([r"\end{document}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_pdf_tables(
    results_by_target: dict[str, list[dict[str, Any]]],
    output_path: Path,
    figures_dir: Path | None = None,
) -> None:
    """PDF con tablas, métricas y, si existe, figuras incrustadas.

    Usa reportlab con fuentes TrueType (soporta acentos), imágenes
    raster y tablas con estilo consistente. Las figuras que se incluyen
    son las que `plots.py` guarda en `figures_dir/experiments/` y
    `figures_dir/analysis/`.
    """
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.fontSize = 9
    body_style.leading = 12

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Laboratorio 04: clasificadores de ensamble",
    )

    story: list[Any] = []
    story.append(Paragraph("Laboratorio 04: clasificadores de ensamble", title_style))
    story.append(
        Paragraph(
            "Bagging, AdaBoost, Stacking y Gradient Boosting evaluados con "
            "validacion cruzada anidada. Se comparan las estrategias "
            r"grid\_all y random\_all, y se reportan F1 macro, balanced "
            "accuracy, ICN crudo y normalizado, estabilidad y "
            "$\\Delta$sesgo.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.5 * cm))

    # Figuras opcionales: heatmap F1 macro, comparativa grid vs random,
    # comparativa entre laboratorios.
    f1_heatmap = figures_dir / "experiments" / "f1_macro_heatmap.png" if figures_dir else None
    grid_vs_random = figures_dir / "analysis" / "grid_vs_random_heatmap.png" if figures_dir else None
    lab3_vs_lab4 = figures_dir / "analysis" / "lab3_vs_lab4_f1.png" if figures_dir else None

    if f1_heatmap is not None and f1_heatmap.exists():
        story.append(Paragraph("F1 macro por modelo y objetivo", h2_style))
        story.append(Image(str(f1_heatmap), width=22 * cm, height=8 * cm))
        story.append(Spacer(1, 0.5 * cm))

    for target, results in results_by_target.items():
        story.append(Paragraph(f"Experimento {target}", h2_style))
        distribution = _format_distribution(results[0]["class_distribution"])
        n_min = results[0]["n_min"]
        k_outer = results[0]["k_outer"]
        story.append(
            Paragraph(
                f"Distribucion: {distribution}. n_min = {n_min}. k externo = {k_outer}.",
                body_style,
            )
        )
        story.append(Spacer(1, 0.3 * cm))
        story.append(_build_reportlab_table(results))
        story.append(PageBreak())

    # Página final con figuras adicionales si existen.
    extra_figures: list[tuple[str, Path, float, float]] = []
    if grid_vs_random is not None and grid_vs_random.exists():
        extra_figures.append(("Grid vs Random (F1 macro)", grid_vs_random, 22, 9))
    if lab3_vs_lab4 is not None and lab3_vs_lab4.exists():
        extra_figures.append(("Comparativa Lab 3 vs Lab 4 (F1 macro)", lab3_vs_lab4, 22, 8))

    for title, path, w, h in extra_figures:
        story.append(Paragraph(title, h2_style))
        story.append(Image(str(path), width=w * cm, height=h * cm))
        story.append(Spacer(1, 0.3 * cm))

    doc.build(story)


def _build_reportlab_table(results: list[dict[str, Any]]) -> Table:
    """Tabla con la misma información que las tablas LaTeX."""
    header = [
        "Experimento / modelo",
        "F1 macro",
        "BalAcc",
        "Recall",
        "Precision",
        "ICN*",
        "ICN",
        "Estab*",
        "Delta sesgo",
        "Best params",
    ]
    rows: list[list[str]] = [header]
    for item in results:
        if not item["implemented"]:
            label = f"{item.get('experiment_name', '')} / {item['model_name']}"
            rows.append([label, "No implementado", *[""] * 8])
            continue
        label = f"{item.get('experiment_name', '')} / {item['model_name']}"
        best_params_short = item["best_params_mode"]
        if len(best_params_short) > 80:
            best_params_short = best_params_short[:77] + "..."
        rows.append(
            [
                label,
                _format_mean_std(item["f1_macro_mean"], item["f1_macro_std"]),
                _format_float(item["balanced_accuracy_mean"]),
                _format_float(item["recall_macro_mean"]),
                _format_float(item["precision_macro_mean"]),
                _format_float(item["icn_raw"]),
                _format_float(item["icn"]),
                _format_float(item["stability_raw"]),
                _format_float(item.get("delta_sesgo")),
                best_params_short,
            ]
        )

    table = Table(
        rows,
        colWidths=[
            4.5 * cm, 2.4 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm,
            1.3 * cm, 1.3 * cm, 1.3 * cm, 1.7 * cm, 5.5 * cm,
        ],
        repeatRows=1,
    )
    style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.0),
            ("ALIGN", (1, 1), (-2, -1), "CENTER"),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ("TOPPADDING", (0, 0), (-1, 0), 4),
        ]
    )
    for row_idx in range(1, len(rows)):
        if row_idx % 2 == 0:
            style.add("BACKGROUND", (0, row_idx), (-1, row_idx), colors.whitesmoke)
    table.setStyle(style)
    return table


def _write_confusion_matrix(
    target: str, item: dict[str, Any], output_dir: Path
) -> None:
    labels = item["labels"]
    matrix = item["confusion_matrix"]
    experiment_name = item.get("experiment_name", "experiment")
    path = (
        output_dir
        / f"matriz_confusion_{experiment_name}_{target}_{item['model_key']}.csv"
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["real/predicho", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def _write_class_report(
    target: str, item: dict[str, Any], output_dir: Path
) -> None:
    report = item["classification_report"]
    experiment_name = item.get("experiment_name", "experiment")
    path = (
        output_dir
        / f"metricas_por_clase_{experiment_name}_{target}_{item['model_key']}.csv"
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clase", "precision", "recall", "f1-score", "support"])
        for label in item["labels"]:
            row = report[str(label)]
            writer.writerow(
                [
                    label,
                    _format_float(row["precision"]),
                    _format_float(row["recall"]),
                    _format_float(row["f1-score"]),
                    int(row["support"]),
                ]
            )


def _latex_row(item: dict[str, Any]) -> str:
    model = latex_escape(
        f"{item.get('experiment_name', '')} / {item['model_name']}"
    )
    if not item["implemented"]:
        status = latex_escape(item["status"])
        return (
            f"{model} & {status} & {status} & {status} & {status} & "
            f"{status} & {status} & --- & --- & --- & "
            f"{latex_escape(item['message'])} \\\\"
        )

    n_iter_cell = (
        str(int(item["search_n_iter"]))
        if item.get("search_n_iter") is not None
        else "---"
    )
    search_type = latex_escape(str(item.get("search_type") or "---"))
    return (
        f"{model} & "
        f"{_format_mean_std(item['f1_macro_mean'], item['f1_macro_std'])} & "
        f"{_format_float(item['balanced_accuracy_mean'])} & "
        f"{_format_float(item['recall_macro_mean'])} & "
        f"{_format_float(item['precision_macro_mean'])} & "
        f"{_format_float(item['icn_raw'])} & "
        f"{_format_float(item['icn'])} & "
        f"{_format_float(item.get('delta_sesgo'))} & "
        f"{search_type} & "
        f"{n_iter_cell} & "
        f"{latex_escape(item['best_params_mode'])} \\\\"
    )


def _format_distribution(distribution: dict[int, int]) -> str:
    return ", ".join(f"{label}: {support}" for label, support in distribution.items())


def _format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "No implementado"
    return f"{mean:.3f} +/- {std:.3f}"


def _format_float(value: float | None) -> str:
    if value is None:
        return "No implementado"
    return f"{value:.3f}"


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return value


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def merge_historical_results(
    new_results_by_target: dict[str, list[dict[str, Any]]], tables_dir: Path
) -> dict[str, list[dict[str, Any]]]:
    """Combina los resultados ejecutados en la corrida actual con el cache
    o manifiesto histórico previo en `resultados_detallados.json`. De esta
    forma, una corrida parcial no borra los modelos ni targets ya calculados.
    """
    json_path = tables_dir / "resultados_detallados.json"
    if not json_path.exists():
        return new_results_by_target

    try:
        with json_path.open("r", encoding="utf-8") as handle:
            historical: dict[str, list[dict[str, Any]]] = json.load(handle)
    except Exception:
        return new_results_by_target

    merged: dict[str, list[dict[str, Any]]] = dict(historical)
    for target, new_list in new_results_by_target.items():
        if target not in merged:
            merged[target] = list(new_list)
            continue

        existing_list = list(merged[target])
        by_key = {
            (item.get("experiment_name", ""), item.get("model_key", "")): idx
            for idx, item in enumerate(existing_list)
        }

        for new_item in new_list:
            run_key = (new_item.get("experiment_name", ""), new_item.get("model_key", ""))
            if run_key in by_key:
                old_item = existing_list[by_key[run_key]]
                raw_old = old_item.get("f1_macro_mean")
                raw_new = new_item.get("f1_macro_mean")
                old_f1 = raw_old if (raw_old is not None and raw_old == raw_old) else -1.0
                new_f1 = raw_new if (raw_new is not None and raw_new == raw_new) else -1.0
                if new_f1 >= old_f1:
                    existing_list[by_key[run_key]] = new_item
            else:
                existing_list.append(new_item)

        merged[target] = existing_list

    return merged
