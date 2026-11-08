"""Lógica experimental y de aprendizaje automático de AEROMAX-IA.

Solo admite mediciones REALES del diseño descrito en el plan.
Las pruebas automatizadas usan datos artificiales identificados como tales.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

BENDS = (0.0, 0.5, 1.0)
CLIPS = (0, 1, 2, 3)
REPETITIONS = (1, 2, 3, 4)
FEATURES = ["doblez_cm", "clips"]
TARGET = "distancia_m"
MODEL_NAMES = ("Promedio (Constant)", "Regresión lineal", "Random Forest")


def experiment_template() -> pd.DataFrame:
    """48 filas: 12 configuraciones por 4 repeticiones, sin resultados inventados."""
    rows: list[dict[str, Any]] = []
    designs = list(product(BENDS, CLIPS))
    # Intercalar diseños entre rondas ayuda a reducir sesgos por cansancio.
    for repetition in REPETITIONS:
        offset = (repetition - 1) * 3
        ordered = designs[offset:] + designs[:offset]
        if repetition % 2 == 0:
            ordered = list(reversed(ordered))
        for bend, clips in ordered:
            rows.append(
                {
                    "vuelo_id": len(rows) + 1,
                    "doblez_cm": bend,
                    "clips": clips,
                    "repeticion": repetition,
                    "distancia_m": np.nan,
                }
            )
    return pd.DataFrame(rows)


def template_as_csv() -> bytes:
    return experiment_template().to_csv(index=False, float_format="%.2f").encode("utf-8-sig")


def read_uploaded_csv(file: Any) -> pd.DataFrame:
    """Acepta CSV con delimitador coma o punto y coma, con decimales coma o punto."""
    payload = file.read() if hasattr(file, "read") else file
    if isinstance(payload, bytes):
        text = payload.decode("utf-8-sig")
    else:
        text = str(payload)
    return pd.read_csv(StringIO(text), sep=None, engine="python", dtype=str)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def normalize_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos sin cambiar los valores ni inventar mediciones."""
    df = frame.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    required = [*FEATURES, TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Faltan columnas: " + ", ".join(missing))
    for c in required:
        df[c] = _numeric(df[c])
    if "repeticion" in df.columns:
        df["repeticion"] = _numeric(df["repeticion"])
    else:
        # Para CSV del plan con solo 3 columnas, el orden determina la repetición.
        df["repeticion"] = df.groupby(FEATURES, dropna=False).cumcount() + 1
    return df


def validate_measurements(frame: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """Valida el protocolo completo antes de habilitar entrenamiento."""
    try:
        df = normalize_measurements(frame)
    except (ValueError, TypeError) as exc:
        return None, [str(exc)]
    errors: list[str] = []
    if len(df) != 48:
        errors.append(f"Se requieren 48 registros; se encontraron {len(df)}.")

    invalid_bend = ~df["doblez_cm"].isin(BENDS)
    invalid_clips = ~df["clips"].isin(CLIPS)
    invalid_repet = ~df["repeticion"].isin(REPETITIONS)
    if invalid_bend.any():
        errors.append("Doblez inválido: solo se admiten 0; 0,5; 1 cm.")
    if invalid_clips.any():
        errors.append("Cantidad de clips inválida: solo se admiten 0, 1, 2 o 3.")
    if invalid_repet.any():
        errors.append("La repetición debe ser 1, 2, 3 o 4 para cada configuración.")

    missing_count = int(df[TARGET].isna().sum())
    if missing_count:
        errors.append(f"Faltan {missing_count} distancias reales por registrar.")
    if df[TARGET].notna().any() and (df[TARGET].dropna() <= 0).any():
        errors.append("Las distancias medidas deben ser mayores que 0 metros.")

    if not (invalid_bend.any() or invalid_clips.any() or invalid_repet.any()):
        expected = set(product(BENDS, CLIPS, REPETITIONS))
        observed = set(zip(df["doblez_cm"], df["clips"], df["repeticion"]))
        if len(df.drop_duplicates([*FEATURES, "repeticion"])) != len(df):
            errors.append("Hay repeticiones duplicadas para una misma configuración.")
        if expected != observed:
            missing = len(expected - observed)
            extra = len(observed - expected)
            errors.append(
                f"Combinaciones/repeticiones incompletas o extra: {missing} faltantes, {extra} extra."
            )
    return (df if not errors else None), errors


def split_experiment(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """El vuelo 4 de cada configuración queda fuera del entrenamiento."""
    train = df.loc[df["repeticion"] != 4].copy()
    test = df.loc[df["repeticion"] == 4].copy()
    if len(train) != 36 or len(test) != 12:
        raise ValueError("División incorrecta: deben ser 36 entrenamientos y 12 pruebas.")
    return train, test


@dataclass
class ExperimentResult:
    train: pd.DataFrame
    test: pd.DataFrame
    models: dict[str, Any]
    errors: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame

    @property
    def best_model(self) -> str:
        return str(self.metrics.iloc[0]["modelo"])


def train_and_evaluate(df: pd.DataFrame) -> ExperimentResult:
    """Entrena tres modelos usando solo 36 registros y evalúa los 12 restantes."""
    normalized, errors = validate_measurements(df)
    if errors or normalized is None:
        raise ValueError("No es posible entrenar: " + " | ".join(errors))
    train, test = split_experiment(normalized)
    models = {
        "Promedio (Constant)": DummyRegressor(strategy="mean"),
        "Regresión lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, min_samples_leaf=1, n_jobs=1
        ),
    }
    summary: list[dict[str, Any]] = []
    predictions = test[[*FEATURES, "repeticion", TARGET]].copy()
    long_errors: list[dict[str, Any]] = []
    for name, model in models.items():
        model.fit(train[FEATURES], train[TARGET])
        pred = model.predict(test[FEATURES])
        predictions[f"pred_{name}"] = pred
        predictions[f"error_{name}"] = np.abs(test[TARGET].to_numpy() - pred)
        mae = float(mean_absolute_error(test[TARGET], pred))
        summary.append({"modelo": name, "MAE_m": mae})
        for idx, real, estimate in zip(test.index, test[TARGET], pred):
            long_errors.append(
                {
                    "vuelo": int(idx) + 1,
                    "modelo": name,
                    "real_m": float(real),
                    "estimado_m": float(estimate),
                    "error_absoluto_m": abs(float(real) - float(estimate)),
                }
            )
    metrics = pd.DataFrame(summary).sort_values(
        ["MAE_m", "modelo"], kind="stable"
    ).reset_index(drop=True)
    return ExperimentResult(
        train=train,
        test=test,
        models=models,
        errors=pd.DataFrame(long_errors),
        predictions=predictions.reset_index(drop=True),
        metrics=metrics,
    )


def predict_all_designs(result: ExperimentResult, model_name: str) -> pd.DataFrame:
    if model_name not in result.models:
        raise ValueError("Modelo desconocido")
    configurations = pd.DataFrame(
        [{"doblez_cm": bend, "clips": clips} for bend, clips in product(BENDS, CLIPS)]
    )
    configurations["estimado_m"] = result.models[model_name].predict(
        configurations[FEATURES]
    )
    return configurations.sort_values(
        ["estimado_m", "doblez_cm", "clips"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def has_unique_optimum(predictions: pd.DataFrame, tolerance: float = 1e-9) -> bool:
    """La igualdad de puntuación en el primer lugar impide recomendar uno solo."""
    values = predictions["estimado_m"].to_numpy(dtype=float)
    if len(values) < 2:
        return False
    return bool((values[0] - values[1]) > tolerance)


def compare_confirmation(
    selected: list[float], reference: list[float]
) -> dict[str, float | bool]:
    """Comparación descriptiva de 5 lanzamientos adicionales por diseño."""
    a = np.asarray(selected, dtype=float)
    b = np.asarray(reference, dtype=float)
    if len(a) != 5 or len(b) != 5 or not (
        np.isfinite(a).all() and np.isfinite(b).all()
    ):
        raise ValueError("Ingresa exactamente 5 distancias válidas para cada diseño.")
    if (a <= 0).any() or (b <= 0).any():
        raise ValueError("Cada distancia debe ser mayor que 0 metros.")
    ma, mb = float(a.mean()), float(b.mean())
    return {
        "promedio_ia_m": ma,
        "promedio_referencia_m": mb,
        "dispersion_ia_m": float(a.std(ddof=1)),
        "dispersion_referencia_m": float(b.std(ddof=1)),
        "diferencia_m": ma - mb,
        "mejora_pct": (ma - mb) / mb * 100,
        "supera_referencia": bool(ma > mb),
    }
