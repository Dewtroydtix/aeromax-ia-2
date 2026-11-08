"""Aplicación web local AEROMAX-IA.

Ejecutar:
    pip install -r requirements.txt
    streamlit run app.py

No utiliza una API externa ni presenta datos simulados como resultados reales.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from aeromax_core import (
    BENDS,
    CLIPS,
    FEATURES,
    MODEL_NAMES,
    TARGET,
    compare_confirmation,
    experiment_template,
    has_unique_optimum,
    normalize_measurements,
    predict_all_designs,
    read_uploaded_csv,
    split_experiment,
    template_as_csv,
    train_and_evaluate,
    validate_measurements,
)
from project_knowledge import SUGGESTIONS, answer_question

st.set_page_config(
    page_title="AEROMAX-IA | EXPOMATEMÁTICA 2026",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.8rem; max-width: 1250px}
      h1, h2, h3 {color: #133453}
      [data-testid="stMetric"] {
        background: #ecf4f8; border: 1px solid #d6e7f0;
        border-radius: 12px; padding: 14px;
      }
      [data-testid="stMetricLabel"] {font-weight: 600}
      .subtle {color:#55677a;font-size:0.98rem}
      .lead {
        background: linear-gradient(105deg,#123452,#176786);
        color:#fff; border-radius:16px; padding:24px 28px;
        margin-bottom:16px;
      }
      .lead h1 {color:#fff;margin:0;font-size:2rem}
      .lead p {margin:7px 0 0;color:#e5f4fa}
      .note {background:#f1f6fa;border-left:4px solid #2582a8;
             padding:12px 16px;border-radius:5px;color:#22445c}
    </style>
    """,
    unsafe_allow_html=True,
)


def _signature(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        df.to_csv(index=False, na_rep="", float_format="%.12g").encode("utf-8")
    ).hexdigest()


def _label_design(bend: float, clips: int) -> str:
    return f"Doblez {bend:g} cm · {clips} clip{'s' if clips != 1 else ''}"


def _local_pdf() -> bytes | None:
    path = Path(__file__).resolve().parent / "PLAN_AEROMAX_IA.pdf"
    return path.read_bytes() if path.exists() else None


if "measurements" not in st.session_state:
    st.session_state.measurements = experiment_template()
if "trained_result" not in st.session_state:
    st.session_state.trained_result = None
if "trained_signature" not in st.session_state:
    st.session_state.trained_signature = ""
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


st.markdown(
    """
    <div class="lead">
      <h1>✈ AEROMAX-IA</h1>
      <p>Inteligencia artificial para predecir y optimizar vuelos de aviones de papel</p>
      <p>EXPOMATEMÁTICA JUANPABLINA 2026 · Categoría B · Álgebra aplicada</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Proyecto de I.E.P. Juan Pablo Peregrino. Las predicciones reales se activan "
    "únicamente después de registrar las 48 mediciones del experimento."
)

with st.sidebar:
    st.title("Panel del proyecto")
    st.markdown("**Diseño experimental**")
    st.write("3 dobleces × 4 cantidades de clips = **12 diseños**.")
    st.write("4 repeticiones por diseño = **48 vuelos**.")
    st.write("Entrenamiento: **36** vuelos.")
    st.write("Prueba independiente: **12** vuelos.")
    st.write("Verificación posterior: **10** vuelos.")
    st.divider()
    st.markdown("**Modelos comparados**")
    st.write("Random Forest · Regresión lineal · Promedio constante")
    st.divider()
    st.markdown("**Información importante**")
    st.info(
        "La IA aprende de distancias medidas, no de respuestas predefinidas. "
        "No hay ganador ni error de predicción hasta disponer de datos reales."
    )
    pdf_data = _local_pdf()
    if pdf_data:
        st.download_button(
            "Descargar plan original (PDF)",
            pdf_data,
            file_name="PLAN_AEROMAX_IA.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

tab_data, tab_models, tab_predict, tab_confirm, tab_assist = st.tabs(
    [
        "1 · Registrar vuelos",
        "2 · Entrenar y evaluar",
        "3 · Optimizar diseño",
        "4 · Confirmar en vuelo",
        "5 · Preguntar a la IA",
    ]
)


with tab_data:
    st.header("Registro de las 48 mediciones reales")
    st.write(
        "Mide la **distancia horizontal** de cada vuelo en metros. "
        "Mantén constantes papel, plegado básico, lanzador y punto de partida."
    )
    left, right = st.columns([2.2, 1])
    with left:
        uploaded = st.file_uploader(
            "Importar un CSV con mediciones",
            type=["csv"],
            help="Columnas obligatorias: doblez_cm, clips, distancia_m. "
            "Se recomiendan también vuelo_id y repeticion.",
        )
        if uploaded is not None:
            incoming = uploaded.getvalue()
            up_sig = hashlib.sha256(incoming).hexdigest()
            if st.session_state.get("last_uploaded_signature") != up_sig:
                try:
                    received = read_uploaded_csv(incoming)
                    # Confirmar nombres de columnas, sin completar valores inexistentes.
                    normalize_measurements(received)
                    st.session_state.measurements = received
                    st.session_state.last_uploaded_signature = up_sig
                    st.session_state.trained_result = None
                    if "data_editor" in st.session_state:
                        del st.session_state["data_editor"]
                    st.success("CSV importado. Comprueba el contenido antes de entrenar.")
                except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
                    st.error(f"El CSV no se pudo importar: {exc}")
    with right:
        st.download_button(
            "Descargar plantilla vacía (48 vuelos)",
            data=template_as_csv(),
            file_name="plantilla_48_vuelos.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("Restablecer registro vacío", use_container_width=True):
            st.session_state.measurements = experiment_template()
            st.session_state.trained_result = None
            st.session_state.last_uploaded_signature = None
            if "data_editor" in st.session_state:
                del st.session_state["data_editor"]
            st.rerun()

    df_for_editor = st.session_state.measurements.copy()
    # Aceptamos CSV de 3 columnas, agregamos repetición inferida para la edición.
    try:
        df_for_editor = normalize_measurements(df_for_editor)
    except (ValueError, TypeError):
        pass
    if all(c in df_for_editor.columns for c in ["doblez_cm", "clips", "repeticion", "distancia_m"]):
        cols = [c for c in ["vuelo_id", "doblez_cm", "clips", "repeticion", "distancia_m"]
                if c in df_for_editor.columns]
        display_df = df_for_editor[cols].copy().reset_index(drop=True)
        st.write("**Escribe únicamente las distancias observadas en la columna editable.**")
        edited = st.data_editor(
            display_df,
            key="data_editor",
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            disabled=[c for c in cols if c != "distancia_m"],
            column_config={
                "vuelo_id": st.column_config.NumberColumn("N.º de vuelo", format="%d"),
                "doblez_cm": st.column_config.NumberColumn("Doblez (cm)", format="%.1f"),
                "clips": st.column_config.NumberColumn("Clips", format="%d"),
                "repeticion": st.column_config.NumberColumn("Repetición", format="%d"),
                "distancia_m": st.column_config.NumberColumn(
                    "Distancia real (m)", min_value=0.01, format="%.2f"
                ),
            },
        )
        st.session_state.measurements = edited
        n_measured = int(pd.to_numeric(edited[TARGET], errors="coerce").notna().sum())
        st.progress(min(n_measured / 48, 1.0), text=f"{n_measured} / 48 distancias registradas")
        st.download_button(
            "Guardar mediciones actuales en CSV",
            edited.to_csv(index=False, float_format="%.3f").encode("utf-8-sig"),
            file_name="aeromax_mediciones.csv",
            mime="text/csv",
        )
    else:
        st.error("El CSV necesita las columnas doblez_cm, clips y distancia_m.")
    _, validation_errors = validate_measurements(st.session_state.measurements)
    if validation_errors:
        with st.expander("Requisitos pendientes para entrenar", expanded=False):
            for issue in validation_errors:
                st.write("• " + issue)
    else:
        st.success("El registro contiene 48 mediciones válidas, con 4 repeticiones por diseño.")


with tab_models:
    st.header("Entrenamiento y comparación de tres modelos")
    st.write(
        "La aplicación usa **Random Forest** como IA principal y lo compara "
        "con regresión lineal y predicción por el promedio. La repetición 4 de "
        "cada diseño queda reservada para prueba: **36 para entrenamiento y 12 para evaluación**."
    )
    current_sig = _signature(st.session_state.measurements)
    _, check_errors = validate_measurements(st.session_state.measurements)
    can_train = not check_errors
    if check_errors:
        st.warning("Para entrenar, completa y valida los 48 registros en la pestaña 1.")
    if st.button(
        "Entrenar los 3 modelos y calcular el MAE",
        type="primary",
        disabled=not can_train,
    ):
        try:
            trained = train_and_evaluate(st.session_state.measurements)
            st.session_state.trained_result = trained
            st.session_state.trained_signature = current_sig
            if "selected_model" in st.session_state:
                del st.session_state["selected_model"]
            st.success("Entrenamiento terminado. Se evaluaron los 12 vuelos reservados.")
        except ValueError as exc:
            st.error(str(exc))

    if (
        st.session_state.trained_result is not None
        and st.session_state.trained_signature != current_sig
    ):
        st.session_state.trained_result = None
        st.warning("Las mediciones cambiaron. Reentrena para actualizar las predicciones.")

    result = st.session_state.trained_result
    if result is None:
        st.info(
            "En cuanto haya 48 distancias reales, aquí aparecerán la tabla de MAE "
            "y los gráficos de predicción. No se incluyen resultados inventados."
        )
    else:
        ordered = result.metrics.copy()
        a, b, c = st.columns(3)
        for col, row in zip((a, b, c), ordered.itertuples()):
            with col:
                st.metric(row.modelo, f"{row.MAE_m:.3f} m", help="MAE: menor es mejor.")
        st.subheader("Comparación en las 12 pruebas independientes")
        st.dataframe(
            ordered.rename(columns={"modelo": "Modelo", "MAE_m": "MAE (m)"}),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Ver la función algebraica ajustada con los datos"):
            linear_model = result.models["Regresión lineal"]
            a = float(linear_model.intercept_)
            b, c = map(float, linear_model.coef_)
            st.latex(
                rf"\hat{{d}} = {a:.3f} {b:+.3f}x {c:+.3f}n"
            )
            st.write(
                f"Coeficientes estimados con las 36 mediciones de entrenamiento: "
                f"a = {a:.3f}, b = {b:.3f}, c = {c:.3f}. "
                "x es el doblez en cm; n es el número de clips; d está en metros."
            )
        scores = dict(zip(ordered["modelo"], ordered["MAE_m"]))
        if scores["Random Forest"] < scores["Promedio (Constant)"]:
            st.success(
                "Random Forest cumple el criterio del plan: tiene MAE menor "
                "que la predicción por promedio en estas 12 pruebas."
            )
        else:
            st.warning(
                "Random Forest NO superó al promedio constante en estas 12 pruebas. "
                "Esto es un resultado válido y debe informarse al jurado."
            )

        chart_name = st.selectbox(
            "Selecciona un modelo para revisar sus 12 predicciones",
            MODEL_NAMES,
            index=list(MODEL_NAMES).index(result.best_model),
            key="chart_model",
        )
        plot_data = pd.DataFrame(
            {
                "Real (m)": result.predictions[TARGET].to_numpy(),
                f"Predicción: {chart_name} (m)": result.predictions[f"pred_{chart_name}"].to_numpy(),
            },
            index=np.arange(1, 13),
        )
        st.line_chart(plot_data)
        st.caption(
            "Cada posición muestra el vuelo 4 de una configuración diferente. "
            "La diferencia entre el valor real y el estimado determina el error."
        )
        with st.expander("Ver predicciones y errores detallados"):
            show = result.predictions[
                ["doblez_cm", "clips", TARGET, f"pred_{chart_name}", f"error_{chart_name}"]
            ].rename(
                columns={
                    TARGET: "Real (m)",
                    f"pred_{chart_name}": "Estimado (m)",
                    f"error_{chart_name}": "Error absoluto (m)",
                }
            )
            st.dataframe(show, hide_index=True, use_container_width=True)

        train_csv = result.train[FEATURES + [TARGET]].to_csv(index=False)
        test_csv = result.test[FEATURES + [TARGET]].to_csv(index=False)
        l, r = st.columns(2)
        with l:
            st.download_button(
                "Descargar entrenamiento.csv (36)",
                train_csv.encode("utf-8-sig"),
                file_name="entrenamiento.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with r:
            st.download_button(
                "Descargar pruebas.csv (12)",
                test_csv.encode("utf-8-sig"),
                file_name="pruebas.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.caption(
            "Estos dos archivos también se pueden importar en Orange Data Mining, "
            "como indica el plan."
        )


with tab_predict:
    st.header("Optimización entre las 12 configuraciones")
    result = st.session_state.trained_result
    if result is None:
        st.info("Primero entrena los modelos con las 48 mediciones reales.")
    else:
        available = list(MODEL_NAMES)
        chosen = st.selectbox(
            "Modelo utilizado para recomendar un diseño",
            available,
            index=available.index(result.best_model),
            key="selected_model",
        )
        if chosen != result.best_model:
            st.info(
                f"Has elegido {chosen}. El modelo con menor MAE en las pruebas "
                f"fue {result.best_model}."
            )
        grid = predict_all_designs(result, chosen)
        display = grid.copy()
        display.columns = ["Doblez (cm)", "Clips", "Distancia estimada (m)"]
        st.dataframe(display, use_container_width=True, hide_index=True)
        bar_data = pd.DataFrame(
            {
                "Configuración": [
                    _label_design(float(row.doblez_cm), int(row.clips))
                    for row in grid.itertuples()
                ],
                "Distancia estimada (m)": grid["estimado_m"].tolist(),
            }
        )
        st.bar_chart(
            bar_data,
            x="Configuración",
            y="Distancia estimada (m)",
        )
        if has_unique_optimum(grid):
            candidate = grid.iloc[0]
            st.success(
                f"**Configuración recomendada por {chosen}:** doblez de "
                f"**{candidate['doblez_cm']:g} cm**, **{int(candidate['clips'])} clips**. "
                f"Distancia estimada: **{candidate['estimado_m']:.2f} m**."
            )
            st.write(
                "Esta es una **predicción, no una medición definitiva**. "
                "Valídala con cinco vuelos nuevos y compárala con una referencia."
            )
        else:
            st.warning(
                "Hay empate en el valor máximo estimado. Este modelo no permite "
                "identificar un único mejor diseño; evita declarar un ganador."
            )
        st.caption(
            "La optimización no se extiende a dobleces o cantidades de clips "
            "fuera de los valores definidos en el proyecto."
        )


with tab_confirm:
    st.header("Verificación con 10 vuelos adicionales")
    st.write(
        "Compara **cinco vuelos nuevos** del diseño elegido con **cinco nuevos** "
        "de una configuración de referencia. No utilices ninguno de los 48 vuelos iniciales."
    )
    result = st.session_state.trained_result
    if result is None:
        st.info("Primero entrena y evalúa los modelos.")
    else:
        current_model = st.session_state.get("selected_model", result.best_model)
        grid = predict_all_designs(result, current_model)
        if not has_unique_optimum(grid):
            st.warning(
                "El modelo elegido tiene un máximo empatado. "
                "Selecciona otro modelo capaz de distinguir una configuración "
                "o explica la ausencia de un ganador."
            )
        else:
            top = grid.iloc[0]
            selected_pair = (float(top["doblez_cm"]), int(top["clips"]))
            alternatives = [
                (float(b), int(n)) for b in BENDS for n in CLIPS
                if (float(b), int(n)) != selected_pair
            ]
            reference_pair = st.selectbox(
                "Selecciona un diseño de referencia",
                alternatives,
                format_func=lambda pair: _label_design(*pair),
                key="reference_design",
            )
            st.write(
                "**Recomendado:** " + _label_design(*selected_pair) + " | "
                "**Referencia:** " + _label_design(*reference_pair)
            )
            key = (selected_pair, reference_pair, current_model)
            if st.session_state.get("confirmation_key") != key:
                st.session_state.confirmation_key = key
                st.session_state.confirmation_data = pd.DataFrame(
                    {
                        "grupo": ["Elegido por IA"] * 5 + ["Referencia"] * 5,
                        "vuelo_nuevo": list(range(1, 6)) * 2,
                        "distancia_m": [np.nan] * 10,
                    }
                )
                if "confirmation_editor" in st.session_state:
                    del st.session_state["confirmation_editor"]
            confirmation = st.data_editor(
                st.session_state.confirmation_data,
                key="confirmation_editor",
                hide_index=True,
                num_rows="fixed",
                disabled=["grupo", "vuelo_nuevo"],
                use_container_width=True,
                column_config={
                    "grupo": "Diseño",
                    "vuelo_nuevo": st.column_config.NumberColumn("Vuelo nuevo", format="%d"),
                    "distancia_m": st.column_config.NumberColumn(
                        "Distancia real (m)", min_value=0.01, format="%.2f"
                    ),
                },
            )
            st.session_state.confirmation_data = confirmation
            st.download_button(
                "Guardar tabla de confirmación",
                confirmation.to_csv(index=False).encode("utf-8-sig"),
                file_name="confirmacion_10_vuelos.csv",
                mime="text/csv",
            )
            if st.button("Calcular comparación de los 10 vuelos", type="primary"):
                try:
                    measured = pd.to_numeric(confirmation["distancia_m"], errors="coerce")
                    values = compare_confirmation(
                        measured.iloc[:5].tolist(),
                        measured.iloc[5:].tolist(),
                    )
                    a, b, c = st.columns(3)
                    a.metric("Promedio elegido", f"{values['promedio_ia_m']:.2f} m")
                    b.metric("Promedio referencia", f"{values['promedio_referencia_m']:.2f} m")
                    c.metric("Diferencia", f"{values['diferencia_m']:+.2f} m")
                    st.write(
                        f"**Dispersión (desviación estándar muestral):** elegido "
                        f"{values['dispersion_ia_m']:.2f} m; referencia "
                        f"{values['dispersion_referencia_m']:.2f} m."
                    )
                    if values["supera_referencia"]:
                        st.success(
                            "En estos cinco vuelos por diseño, el promedio del "
                            "avión seleccionado superó al de referencia."
                        )
                    else:
                        st.warning(
                            "En estas mediciones, el promedio del avión seleccionado "
                            "no superó al de referencia."
                        )
                    st.caption(
                        "Esta comparación es descriptiva. Cinco repeticiones por diseño "
                        "no garantizan que la diferencia sea generalizable."
                    )
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))


with tab_assist:
    st.header("Asistente para explicar y defender el proyecto")
    st.write(
        "Haz preguntas sobre la función matemática, variables, Random Forest, "
        "método experimental, errores o la exposición ante el jurado."
    )
    st.info(
        "El asistente de preguntas utiliza respuestas verificadas del PDF y "
        "los resultados calculados en esta aplicación. **No es un chatbot "
        "generativo**; la IA que aprende a predecir es Random Forest."
    )

    choice = st.selectbox("Preguntas sugeridas", SUGGESTIONS, key="faq_select")
    if st.button("Responder pregunta sugerida"):
        response, source = answer_question(choice, st.session_state.trained_result)
        st.session_state.qa_history.append((choice, response, source))

    with st.form("qa_form", clear_on_submit=True):
        query = st.text_input("Escribe tu pregunta", placeholder="Ej.: ¿Cómo se calcula el MAE?")
        ask = st.form_submit_button("Preguntar", type="primary")
    if ask:
        response, source = answer_question(query, st.session_state.trained_result)
        st.session_state.qa_history.append((query, response, source))

    if not st.session_state.qa_history:
        st.caption("Las respuestas aparecen aquí junto con la página o fuente.")
    for question, response, source in reversed(st.session_state.qa_history[-10:]):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(response)
            st.caption(f"Fuente: {source}")
    if st.session_state.qa_history:
        if st.button("Limpiar conversación"):
            st.session_state.qa_history = []
            st.rerun()

st.divider()
st.caption(
    "AEROMAX-IA · Protocolo del plan institucional (5 páginas) · "
    "Sin resultados predeterminados · Realiza lanzamientos en un lugar interior despejado "
    "y bajo supervisión docente."
)
