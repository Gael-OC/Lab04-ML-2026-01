import json
import pytest
import warnings
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import assign_icn, run_nested_cv
from src.reports import merge_historical_results
from src.models import RobustStackingClassifier, ModelSpec
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def test_icn_independence():
    """Verifica que dos corridas con el mismo model_key pero diferente experiment_name

    no colisionen al normalizar métricas para ICN.
    """
    results = [
        {
            "implemented": True,
            "model_key": "bagging",
            "model_name": "Bagging",
            "experiment_name": "grid_all",
            "f1_macro_mean": 0.80,
            "balanced_accuracy_mean": 0.80,
            "recall_macro_mean": 0.80,
            "precision_macro_mean": 0.80,
            "f1_macro_std": 0.05,
            "stability_raw": 0.95,
        },
        {
            "implemented": True,
            "model_key": "bagging",
            "model_name": "Bagging",
            "experiment_name": "random_all",
            "f1_macro_mean": 0.90,
            "balanced_accuracy_mean": 0.90,
            "recall_macro_mean": 0.90,
            "precision_macro_mean": 0.90,
            "f1_macro_std": 0.02,
            "stability_raw": 0.98,
        },
    ]
    assign_icn(results)
    
    # El f1_macro_std de random_all (0.02) es menor que el de grid_all (0.05),
    # por lo que en stability (1 - normalizado), random_all debe tener 1.0 y grid_all 0.0.
    assert results[0]["stability"] == pytest.approx(0.0, abs=1e-9), f"Grid_all debió obtener stability 0.0, obtuve {results[0]['stability']}"
    assert results[1]["stability"] == pytest.approx(1.0, abs=1e-9), f"Random_all debió obtener stability 1.0, obtuve {results[1]['stability']}"
    assert results[0]["icn"] != results[1]["icn"], "Los ICN no deben ser idénticos tras la normalización"


def test_historical_merge(tmp_path: Path):
    """Verifica que merge_historical_results conserve targets y modelos previos no re-ejecutados."""
    json_path = tmp_path / "resultados_detallados.json"
    historical_data = {
        "GDS": [
            {"experiment_name": "grid_all", "model_key": "adaboost", "f1_macro_mean": 0.5},
            {"experiment_name": "grid_all", "model_key": "bagging", "f1_macro_mean": 0.6},
        ],
        "GDS_R1": [
            {"experiment_name": "grid_all", "model_key": "adaboost", "f1_macro_mean": 0.7},
        ],
    }
    json_path.write_text(json.dumps(historical_data), encoding="utf-8")

    # Simulamos que en esta corrida solo ejecutamos adaboost sobre GDS
    new_results = {
        "GDS": [
            {"experiment_name": "grid_all", "model_key": "adaboost", "f1_macro_mean": 0.8},  # actualizado
        ]
    }

    merged = merge_historical_results(new_results, tmp_path)
    
    # Debe conservar GDS_R1 intacto
    assert "GDS_R1" in merged
    assert len(merged["GDS_R1"]) == 1
    assert merged["GDS_R1"][0]["f1_macro_mean"] == 0.7

    # En GDS, adaboost debe actualizarse a 0.8 y bagging debe mantenerse en 0.6
    gds_models = {item["model_key"]: item["f1_macro_mean"] for item in merged["GDS"]}
    assert gds_models["adaboost"] == 0.8
    assert gds_models["bagging"] == 0.6
    assert len(merged["GDS"]) == 2


def test_stacking_oof_fallback_warning():
    """Verifica que RobustStackingClassifier emita un UserWarning cuando min_count < 2 o < cv."""
    X = np.random.RandomState(42).randn(16, 4)
    y = np.array([0]*15 + [1])  # Clase 1 tiene solo 1 muestra (min_count < 2)

    stacking = RobustStackingClassifier(cv=3, random_state=42)
    with pytest.warns(UserWarning, match="Stacking OOF fallback"):
        stacking.fit(X, y)


def test_stacking_geometry_parameterization():
    """Verifica que RobustStackingClassifier acepte y utilice geometry y final_class_weight."""
    X = np.random.RandomState(42).randn(20, 4)
    y = np.array([0]*10 + [1]*10)

    stacking = RobustStackingClassifier(cv=3, random_state=42, geometry="manhattan", final_class_weight=None)
    stacking.fit(X, y)
    preds = stacking.predict(X)
    assert len(preds) == 20
    assert stacking.final_estimator_.class_weight is None
    # Verificar que el KNN interno tenga metric="manhattan"
    knn_pipe = dict(stacking.estimators_)["knn"]
    assert knn_pipe.named_steps["clf"].metric == "manhattan"


if __name__ == "__main__":
    import tempfile
    print("Ejecutando test_icn_independence...")
    test_icn_independence()
    print("✓ test_icn_independence pasó exitosamente.")

    print("Ejecutando test_historical_merge...")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_historical_merge(Path(tmpdir))
    print("✓ test_historical_merge pasó exitosamente.")

    print("Ejecutando test_stacking_oof_fallback_warning...")
    test_stacking_oof_fallback_warning()
    print("✓ test_stacking_oof_fallback_warning pasó exitosamente.")

    print("Ejecutando test_stacking_geometry_parameterization...")
    test_stacking_geometry_parameterization()
    print("✓ test_stacking_geometry_parameterization pasó exitosamente.")

    print("\n¡TODAS LAS PRUEBAS DE INFRAESTRUCTURA PASARON LIMPIAMENTE!")
