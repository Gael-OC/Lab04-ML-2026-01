from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from src.data_loader import TARGETS, load_sav_dataset, prepare_xy
from src.evaluation import (
    assign_icn,
    compute_delta_sesgo,
    compute_outer_folds,
    run_nested_cv,
    unimplemented_result,
)
from src.models import build_model_registry
from src.reports import (
    merge_historical_results,
    write_auxiliary_tables,
    write_json_results,
    write_latex_tables,
    write_pdf_tables,
    write_summary_csv,
    write_warnings,
)
from src.settings import DEFAULT_CONFIG_PATH, ensure_output_dirs, load_config


def parse_args() -> ArgumentParser:
    """CLI del laboratorio.

    - `--config`: ruta al YAML.
    - `--targets`: subset de los seis objetivos.
    - `--models`: subset del orden del experimento.
    - `--experiments`: subset de {grid_all, random_all}.
    - `--keep-estimators`: persiste los estimadores por fold en
      `outputs/estimator_cache/` para tests de significancia.
    - `--skip-*`: salta EDA, analisis avanzado o plots.
    """
    parser = ArgumentParser(
        description="Ejecuta el Laboratorio 04 con clasificadores de ensamble."
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help="Ruta al archivo YAML de configuración.",
    )
    parser.add_argument(
        "--targets", nargs="*", default=None,
        help="Objetivos a ejecutar. Por defecto usa los seis del YAML.",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Modelos a ejecutar. Por defecto usa el orden del YAML.",
    )
    parser.add_argument(
        "--experiments", nargs="*", default=None,
        help="Experimentos a ejecutar. Por defecto grid_all y random_all.",
    )
    parser.add_argument(
        "--keep-estimators", action="store_true",
        help="Guarda estimadores por fold para tests de significancia sin reentrenar.",
    )
    parser.add_argument(
        "--skip-eda", action="store_true", help="Salta el analisis exploratorio."
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Salta el analisis avanzado (learning curves, importance, etc.).",
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Salta las visualizaciones."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    experiment_config = config["experiment"]
    validation_config = experiment_config["validation"]
    random_state_config = experiment_config["random_state"]
    hyperparameter_config = config["hyperparameter_search"]

    target_names = args.targets or experiment_config.get("targets", TARGETS)
    model_registry = build_model_registry(
        random_state=_random_state(random_state_config, "model")
    )
    experiment_runs = _selected_experiments(
        requested_experiments=args.experiments,
        configured_experiments=experiment_config["experiments"],
    )

    df = load_sav_dataset(config["dataset"]["path"])
    results_by_target: dict[str, list[dict[str, Any]]] = {
        target_name: [] for target_name in target_names
    }

    estimator_cache_dir = (
        Path(config["outputs"]["estimator_cache"])
        if args.keep_estimators and "estimator_cache" in config["outputs"]
        else None
    )

    for experiment_run in experiment_runs:
        experiment_name = experiment_run["name"]
        search_type = experiment_run["search_type"]
        model_order = _selected_models(
            requested_models=args.models or experiment_run["models"],
            model_registry=model_registry,
            search_models=hyperparameter_config[search_type]["models"],
        )

        for target_name in target_names:
            X, y = prepare_xy(df, target_name)
            n_min, k_outer = compute_outer_folds(
                y, validation_config["max_outer_folds"]
            )
            distribution = {
                int(label): int(count)
                for label, count in y.value_counts().sort_index().items()
            }

            for model_key in model_order:
                spec = model_registry[model_key]
                if spec.implemented:
                    search_config = _search_config_for_model(
                        model_key=model_key,
                        search_type=search_type,
                        experiment_run=experiment_run,
                        hyperparameter_config=hyperparameter_config,
                        validation_config=validation_config,
                    )
                    result = run_nested_cv(
                        X,
                        y,
                        target_name,
                        spec,
                        validation_config,
                        search_config,
                        random_state_config,
                        experiment_name=experiment_name,
                        return_estimators=args.keep_estimators,
                        estimator_cache_dir=estimator_cache_dir,
                    )
                    if isinstance(result, tuple):
                        result = result[0]
                else:
                    result = unimplemented_result(
                        target_name=target_name,
                        model_spec=spec,
                        distribution=distribution,
                        n_min=n_min,
                        k_outer=k_outer,
                        experiment_name=experiment_name,
                    )
                results_by_target[target_name].append(result)

    output_dirs = {name: Path(path) for name, path in config["outputs"].items()}
    tables_dir = output_dirs["tables"]
    figures_dir = output_dirs.get("figures", tables_dir.parent / "figures")

    results_by_target = merge_historical_results(results_by_target, tables_dir)

    # ICN se calcula al final porque compara corridas (modelo + experimento).
    for target_results in results_by_target.values():
        assign_icn(target_results)
        compute_delta_sesgo(target_results)

    write_summary_csv(results_by_target, tables_dir / "resumen_resultados.csv")
    write_json_results(results_by_target, tables_dir / "resultados_detallados.json")
    write_auxiliary_tables(results_by_target, output_dirs)
    write_warnings(results_by_target, output_dirs["root"] / "advertencias.txt")
    write_latex_tables(results_by_target, tables_dir / "resultados_experimentos.tex")
    write_pdf_tables(
        results_by_target,
        tables_dir / "resultados_experimentos.pdf",
        figures_dir=figures_dir,
    )

    if not args.skip_analysis:
        from src.analysis import run_advanced_analysis
        analysis_dir = figures_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        run_advanced_analysis(
            results_by_target=results_by_target,
            df=df,
            output_dirs={"root": output_dirs["root"], "tables": tables_dir, "analysis": analysis_dir, "figures": figures_dir},
            estimator_cache_dir=estimator_cache_dir,
        )

    if not args.skip_plots:
        from src.plots import generate_all_plots
        plots_dir = figures_dir / "experiments"
        plots_dir.mkdir(parents=True, exist_ok=True)
        generate_all_plots(results_by_target, plots_dir)

    if not args.skip_eda:
        from src.eda import run_eda
        eda_dir = figures_dir / "eda"
        eda_dir.mkdir(parents=True, exist_ok=True)
        run_eda(df, eda_dir)

    print("Experimentos finalizados.")
    print(f"Tabla LaTeX: {tables_dir / 'resultados_experimentos.tex'}")
    print(f"Tabla PDF:   {tables_dir / 'resultados_experimentos.pdf'}")


def _selected_experiments(
    requested_experiments: list[str] | None,
    configured_experiments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not requested_experiments:
        return configured_experiments

    by_name = {experiment["name"]: experiment for experiment in configured_experiments}
    selected: list[dict[str, Any]] = []
    for experiment_name in requested_experiments:
        if experiment_name not in by_name:
            available = ", ".join(sorted(by_name))
            raise ValueError(
                f"Experimento no reconocido: {experiment_name}. "
                f"Disponibles: {available}."
            )
        selected.append(by_name[experiment_name])
    return selected


def _selected_models(
    requested_models: list[str],
    model_registry: dict[str, Any],
    search_models: dict[str, Any],
) -> list[str]:
    selected: list[str] = []
    for model_key in requested_models:
        if model_key not in model_registry:
            available = ", ".join(sorted(model_registry))
            raise ValueError(
                f"Modelo no reconocido: {model_key}. Disponibles: {available}."
            )
        if model_key not in search_models:
            raise ValueError(
                f"Falta configuración de hiperparámetros para el modelo {model_key}."
            )
        if search_models[model_key].get("enabled", True):
            selected.append(model_key)

    if not selected:
        raise ValueError("No hay modelos habilitados para ejecutar.")
    return selected


def _search_config_for_model(
    model_key: str,
    search_type: str,
    experiment_run: dict[str, Any],
    hyperparameter_config: dict[str, Any],
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    default_config = hyperparameter_config.get("default", {})
    model_config = hyperparameter_config[search_type]["models"][model_key]
    merged: dict[str, Any] = {**default_config, **experiment_run, **model_config}
    merged["type"] = search_type
    merged["scoring"] = merged.get(
        "scoring", validation_config.get("scoring", "f1_macro")
    )
    if search_type == "random":
        merged["n_iter"] = int(
            merged.get("n_iter", default_config.get("random_n_iter", 10))
        )
    return merged


def _random_state(random_state_config: dict[str, int], key: str) -> int:
    if key in random_state_config:
        return int(random_state_config[key])
    return int(random_state_config.get("global", 42))


if __name__ == "__main__":
    main()
