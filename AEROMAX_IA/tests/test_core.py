"""Tests de consistencia. Mediciones artificiales solo para verificar software."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aeromax_core import (
    compare_confirmation,
    experiment_template,
    has_unique_optimum,
    normalize_measurements,
    predict_all_designs,
    split_experiment,
    train_and_evaluate,
    validate_measurements,
)
from project_knowledge import answer_question


def synthetic_for_test() -> pd.DataFrame:
    """Datos ARTIFICIALES para tests; no son resultados del proyecto."""
    frame = experiment_template()
    distances = []
    for row in frame.itertuples():
        # Distancia fabricada para las pruebas del programa exclusivamente.
        distances.append(
            3.2 + 1.7 * row.doblez_cm - 0.35 * row.clips
            + 0.02 * row.repeticion
            + 0.07 * np.sin(row.vuelo_id)
        )
    frame["distancia_m"] = distances
    return frame


class AeromaxCoreTest(unittest.TestCase):
    def test_protocol_has_48_rows_and_no_distances(self):
        df = experiment_template()
        self.assertEqual(len(df), 48)
        self.assertTrue(df["distancia_m"].isna().all())
        self.assertEqual(df.groupby(["doblez_cm", "clips"]).size().nunique(), 1)
        self.assertTrue((df.groupby(["doblez_cm", "clips"]).size() == 4).all())
        # En cada ronda se mezclan las 12 configuraciones.
        self.assertTrue((df.groupby("repeticion").size() == 12).all())

    def test_cannot_train_without_real_measured_distances(self):
        cleaned, errors = validate_measurements(experiment_template())
        self.assertIsNone(cleaned)
        self.assertTrue(any("48 distancias" in e for e in errors))
        with self.assertRaises(ValueError):
            train_and_evaluate(experiment_template())

    def test_three_models_train_and_evaluate_on_12_heldout(self):
        df = synthetic_for_test()
        result = train_and_evaluate(df)
        self.assertEqual(len(result.train), 36)
        self.assertEqual(len(result.test), 12)
        self.assertEqual(len(result.metrics), 3)
        self.assertEqual(len(result.errors), 36)
        self.assertTrue((result.metrics["MAE_m"] >= 0).all())
        grid = predict_all_designs(result, "Random Forest")
        self.assertEqual(len(grid), 12)
        self.assertTrue(np.isfinite(grid["estimado_m"]).all())

    def test_held_out_distance_does_not_leak_into_fit(self):
        original = synthetic_for_test()
        r1 = train_and_evaluate(original)
        changed = original.copy()
        changed.loc[changed["repeticion"] == 4, "distancia_m"] += 100.0
        r2 = train_and_evaluate(changed)
        for model in r1.models:
            p1 = r1.models[model].predict(r1.test[["doblez_cm", "clips"]])
            p2 = r2.models[model].predict(r2.test[["doblez_cm", "clips"]])
            np.testing.assert_allclose(p1, p2, atol=1e-12)
        self.assertTrue(
            (r2.metrics.set_index("modelo")["MAE_m"].sort_index()
             > r1.metrics.set_index("modelo")["MAE_m"].sort_index()).all()
        )

    def test_invalid_duplicate_repetition(self):
        df = synthetic_for_test()
        df.loc[df.index[12], "repeticion"] = 1
        _, errors = validate_measurements(df)
        self.assertTrue(errors)
        self.assertTrue(any("duplicadas" in e for e in errors))

    def test_decimal_comma_is_accepted(self):
        df = synthetic_for_test()
        df["doblez_cm"] = df["doblez_cm"].astype(str).str.replace(".", ",", regex=False)
        df["distancia_m"] = df["distancia_m"].round(2).astype(str).str.replace(
            ".", ",", regex=False
        )
        norm = normalize_measurements(df)
        self.assertEqual(norm["distancia_m"].isna().sum(), 0)
        self.assertEqual(norm["doblez_cm"].nunique(), 3)

    def test_constant_has_tied_maximum(self):
        result = train_and_evaluate(synthetic_for_test())
        grid = predict_all_designs(result, "Promedio (Constant)")
        self.assertFalse(has_unique_optimum(grid))

    def test_confirmation_five_vs_five(self):
        data = compare_confirmation([5, 5.2, 5.1, 5.3, 5.0], [4, 4.2, 4.1, 4.3, 4.0])
        self.assertTrue(data["supera_referencia"])
        self.assertAlmostEqual(data["diferencia_m"], 1.0)
        with self.assertRaises(ValueError):
            compare_confirmation([1, 2], [1, 2, 3, 4, 5])

    def test_assistant_does_not_fabricate_mae(self):
        answer, _ = answer_question("¿Cuál fue el MAE real?")
        self.assertIn("Todavía no hay", answer)
        answer2, _ = answer_question("¿Qué avión es el mejor?")
        self.assertIn("faltan las mediciones", answer2)
        answer3, source = answer_question("¿Cuáles son las variables del proyecto?")
        self.assertIn("doblez", answer3)
        self.assertIn("Página", source)


if __name__ == "__main__":
    unittest.main()
