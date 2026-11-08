"""Respuestas verificables del plan de proyecto (5 páginas).

Este módulo NO simula un modelo generativo: recupera respuestas verificadas del
plan y, cuando hay datos medidos, permite consultar los resultados calculados.
El modelo de IA predictiva sí aprende de las mediciones.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from aeromax_core import ExperimentResult, has_unique_optimum, predict_all_designs

QA = [
    {
        "question": "¿Cuál es el título y qué busca el proyecto AEROMAX-IA?",
        "keywords": "titulo tema objetivo principal inteligencia artificial predecir distancia alcance proyecto",
        "answer": (
            "El proyecto se titula «AEROMAX-IA: ¿Puede una inteligencia artificial predecir y "
            "optimizar la distancia de vuelo de un avión de papel a partir de sus características "
            "de diseño?». Busca desarrollar y evaluar un sistema predictivo que ayude a escoger, "
            "entre las configuraciones estudiadas, un diseño de mayor alcance estimado."
        ),
        "page": "Páginas 1 y 2",
    },
    {
        "question": "¿Cuál es la pregunta de investigación y la hipótesis?",
        "keywords": "problema hipotesis justificacion investigacion alcance precision",
        "answer": (
            "Pregunta: ¿Puede una IA entrenada con datos experimentales predecir la distancia de "
            "vuelo y ayudar a seleccionar una configuración de mayor alcance? Hipótesis: un "
            "modelo de aprendizaje automático podrá predecir y orientar la selección, con "
            "precisión dependiente de la calidad y cantidad de registros. Debe contrastarse "
            "contra regresión lineal y predicción por promedio."
        ),
        "page": "Página 1",
    },
    {
        "question": "¿Qué variables usa la IA y qué valores se permiten?",
        "keywords": "variables entradas doblez cm x alas clips numero n salida d distancia metros",
        "answer": (
            "Entradas: elevación del doblez en la punta de cada ala, x = 0; 0,5; 1 cm, y número "
            "de clips delanteros, n = 0; 1; 2; 3. Salida: distancia horizontal de vuelo, d, "
            "medida en metros. Se estudia d = f(x, n)."
        ),
        "page": "Página 2",
    },
    {
        "question": "¿Cuál es la función o modelo algebraico de comparación?",
        "keywords": "formula ecuacion algebra lineal a b c f x n coeficientes regresion",
        "answer": (
            "El modelo algebraico de comparación es d estimada = a + b·x + c·n. "
            "Los coeficientes a, b y c se calculan con datos observados: no se establecen "
            "ni inventan antes de medir."
        ),
        "page": "Página 2",
    },
    {
        "question": "¿Por qué hay 12 configuraciones de aviones?",
        "keywords": "12 combinaciones configuraciones tres cuatro conteo producto cartesiano",
        "answer": (
            "Hay 3 niveles de doblez (0, 0,5, 1 cm) y 4 cantidades de clips (0, 1, 2, 3). "
            "Por el principio multiplicativo, 3 × 4 = 12 configuraciones."
        ),
        "page": "Páginas 2 y 3",
    },
    {
        "question": "¿Por qué se realizan 48 vuelos?",
        "keywords": "48 cuatro repeticiones vuelos pruebas experimentales cantidad mediciones",
        "answer": (
            "Cada una de las 12 configuraciones se prueba 4 veces: 12 × 4 = 48 "
            "mediciones iniciales. Las repeticiones permiten observar la variación de "
            "los lanzamientos."
        ),
        "page": "Página 3",
    },
    {
        "question": "¿Cómo se dividen los datos para entrenamiento y prueba?",
        "keywords": "36 12 reserva separar entrenamiento prueba test entrenamiento.csv pruebas.csv cuarto vuelo",
        "answer": (
            "Se reservan 3 vuelos de cada configuración para entrenar (12 × 3 = 36) "
            "y 1 para prueba independiente (12 × 1 = 12). La aplicación utiliza la "
            "repetición n.º 4 como prueba y no la incluye en el ajuste de los modelos."
        ),
        "page": "Página 3",
    },
    {
        "question": "¿Qué es Random Forest y por qué se usa?",
        "keywords": "random forest bosque aleatorio arboles aprendizaje automatico modelo principal",
        "answer": (
            "Random Forest es un algoritmo de aprendizaje supervisado de regresión. "
            "Combina varios árboles entrenados con ejemplos medidos para estimar "
            "distancias de vuelo. Es el modelo de IA principal del proyecto, pero no "
            "tiene garantizado superar a modelos más sencillos."
        ),
        "page": "Páginas 2 y 5",
    },
    {
        "question": "¿Con qué modelos se compara la inteligencia artificial?",
        "keywords": "tres modelos baseline constant promedio regresion lineal forest comparacion",
        "answer": (
            "Se comparan tres modelos usando los mismos 12 vuelos de prueba: "
            "Constant (predice el promedio de entrenamiento), regresión lineal "
            "(d = a + b·x + c·n) y Random Forest. Se debe calcular su MAE."
        ),
        "page": "Páginas 2 y 5",
    },
    {
        "question": "¿Qué es el error absoluto y el MAE?",
        "keywords": "mae error absoluto medio formula medir precision diferencia 4,2 3,8",
        "answer": (
            "Error absoluto de un vuelo = |distancia real − distancia estimada|. "
            "MAE = suma de los 12 errores absolutos ÷ 12. Un MAE más bajo indica "
            "predicciones más próximas a esas mediciones de prueba. El ejemplo del "
            "plan (4,2 y 3,8 m) es ilustrativo, NO una medición real."
        ),
        "page": "Página 5",
    },
    {
        "question": "¿Cómo se determina si Random Forest fue exitoso?",
        "keywords": "criterio exito forest comparacion supero promedio mejor mae",
        "answer": (
            "Según el plan, Random Forest logra éxito predictivo comparativo si "
            "su MAE es inferior al de Constant en las 12 pruebas reservadas. "
            "También hay que mostrar el MAE de regresión lineal; la IA podría "
            "no ganarle con un conjunto de entrenamiento pequeño."
        ),
        "page": "Página 5",
    },
    {
        "question": "¿Cómo encuentra la IA la mejor configuración?",
        "keywords": "optimizar optimo configuracion distancia prediccion maximizar alcance diseño ganador",
        "answer": (
            "El modelo seleccionado calcula la distancia estimada para las 12 "
            "configuraciones permitidas y se identifica la de mayor predicción. "
            "Si hay empate, no existe una elección única. Es una recomendación "
            "basada en el modelo: debe comprobarse con vuelos nuevos."
        ),
        "page": "Páginas 2, 3 y 5",
    },
    {
        "question": "¿Cómo se verifica la optimización con nuevos vuelos?",
        "keywords": "validacion confirmacion 10 cinco cinco referencia comprobar mejora dispersion promedio",
        "answer": (
            "Después de recomendar un diseño, se realizan 5 vuelos adicionales "
            "de ese diseño y 5 de una configuración de referencia. Se comparan "
            "las medias y la dispersión. Estos 10 vuelos son distintos de "
            "las 48 mediciones iniciales."
        ),
        "page": "Página 5",
    },
    {
        "question": "¿Qué condiciones experimentales deben mantenerse controladas?",
        "keywords": "controles papel lanzador punto partida orden cansancio interior seguridad",
        "answer": (
            "Mantener igual el tamaño y tipo de papel, el modelo básico de plegado, "
            "el lanzador y el punto de partida. Alternar el orden de los diseños "
            "para reducir el efecto del cansancio. Hacer los vuelos en un espacio "
            "interior despejado y bajo supervisión docente."
        ),
        "page": "Página 3",
    },
    {
        "question": "¿Qué programa gratuito propone el proyecto?",
        "keywords": "orange data mining software programa herramienta codigo bloques visuales file select columns test score predictions",
        "answer": (
            "El plan propone Orange Data Mining como herramienta gratuita y sin "
            "programación. Su flujo incluye File, Select Columns, Random Forest, "
            "Linear Regression, Constant, Test and Score y Predictions. "
            "Esta aplicación proporciona una alternativa programada con Python."
        ),
        "page": "Página 2",
    },
    {
        "question": "¿Cuánto cuesta el proyecto?",
        "keywords": "presupuesto 43 soles costo materiales papel clips cinta metrica regla plumones",
        "answer": (
            "El presupuesto estimado es S/ 43: papel A4 (S/ 5), clips (S/ 3), "
            "cinta adhesiva (S/ 4), cinta métrica (S/ 10), regla (S/ 5), "
            "plumones (S/ 5), cartulina (S/ 7) y hojas de registro (S/ 4). "
            "La computadora prestada y Orange se presupuestan en S/ 0."
        ),
        "page": "Página 4",
    },
    {
        "question": "¿Cómo se organiza la exposición de 8 minutos?",
        "keywords": "jurado exposicion presentacion ocho minutos oral defensa tiempo",
        "answer": (
            "El plan distribuye 8 minutos: 1 para problema y objetivos; "
            "2 para variables, función algebraica y diseño experimental; "
            "2 para entrenamiento y demostración de IA; "
            "2 para errores, comparación y optimización; "
            "1 para conclusiones y preguntas."
        ),
        "page": "Página 4",
    },
    {
        "question": "¿Qué conclusiones y limitaciones tiene el proyecto?",
        "keywords": "conclusion limitacion debilidad recomendaciones 36 pocos datos lanzamiento manual alcance",
        "answer": (
            "Las conclusiones definitivas deben basarse en las mediciones reales. "
            "Las limitaciones incluyen variabilidad del lanzamiento manual y "
            "solo 36 ejemplos de entrenamiento. La optimización se restringe a "
            "las 12 configuraciones estudiadas; no demuestra cuál es el mejor "
            "avión de papel del mundo."
        ),
        "page": "Página 5",
    },
    {
        "question": "¿Cómo podría mejorarse el proyecto?",
        "keywords": "mejoras aumentar repeticiones nuevos disenos variables futuro mejorar datos",
        "answer": (
            "El plan recomienda aumentar las repeticiones, probar nuevas "
            "configuraciones e incorporar más variables sin perder el control "
            "de las condiciones experimentales."
        ),
        "page": "Página 5",
    },
    {
        "question": "¿Qué diferencia hay entre datos reales y ejemplos?",
        "keywords": "resultados reales inventados simulados mediciones pendientes experimental datos",
        "answer": (
            "No deben presentarse distancias inventadas como hallazgos. El plan "
            "incluye un ejemplo de cálculo del error, pero NO resultados "
            "experimentales. Antes de medir no es posible afirmar un MAE real "
            "ni la configuración ganadora."
        ),
        "page": "Páginas 1 y 5",
    },
]

SUGGESTIONS = [
    "¿Qué variables usa la IA?",
    "¿Por qué hay 48 vuelos?",
    "¿Qué es Random Forest?",
    "¿Qué es el MAE?",
    "¿Cómo se comprueba el mejor avión?",
    "¿Qué digo al jurado sobre las limitaciones?",
]


def _clean(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in folded if not unicodedata.combining(c))


# Sin detener palabras comunes, preguntas sin relación con el PDF pueden
# parecer similares solo por "qué", "cómo" o "el proyecto".
SPANISH_STOPWORDS = [
    "que", "como", "cual", "cuales", "quien", "quienes", "por", "porque",
    "cuando", "donde", "para", "con", "sin", "el", "la", "los", "las",
    "un", "una", "unos", "unas", "de", "del", "al", "en", "es", "son",
    "se", "lo", "y", "o", "a", "su", "sus", "mi", "tu", "tiene", "hace",
    "tenemos", "puede", "podemos", "debo", "utilizan", "usar", "usa",
    "proyecto", "aeromax", "ia", "expomatematica",
]
_vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    stop_words=SPANISH_STOPWORDS,
    ngram_range=(1, 2),
    sublinear_tf=True,
)
_corpus = [" ".join((q["question"], q["keywords"])) for q in QA]
_matrix = _vectorizer.fit_transform(_corpus)


def answer_question(
    question: str, result: Optional[ExperimentResult] = None
) -> tuple[str, str]:
    """Devuelve (respuesta, fundamento). Nunca crea resultados experimentales."""
    text = _clean(question).strip()
    if not text:
        return "Escribe una pregunta relacionada con AEROMAX-IA.", "Asistente"
    words = set(re.findall(r"\w+", text))
    # Consultas muy habituales: resolver explícitamente evita confundir
    # "variables del proyecto" con "incorporar nuevas variables".
    is_score = bool(words & {"mae", "error", "errores", "precision"})
    asks_result = bool(
        words & {"cuanto", "cual", "obtuvo", "salio", "resultado", "resultados", "actual", "real"}
    )
    if is_score and asks_result:
        if result is None:
            return (
                "Todavía no hay valores de MAE experimentales. Para calcularlos, "
                "ingresa los 48 vuelos medidos y entrena los tres modelos.",
                "Plan, página 5; estado de la aplicación",
            )
        lines = [
            f"{r.modelo}: MAE = {r.MAE_m:.3f} m"
            for r in result.metrics.itertuples()
        ]
        return (
            "MAE en los 12 vuelos de prueba:\n\n" + "\n\n".join(lines),
            "Resultados calculados con tus mediciones",
        )
    asks_best = bool(words & {"mejor", "optimo", "ganador", "maximo"})
    asks_design = bool(words & {"avion", "diseno", "configuracion", "vuela", "vuelo", "clips", "doblez"})
    if asks_best and asks_design:
        if result is None:
            return (
                "Todavía no se puede saber qué configuración es mejor: faltan las mediciones "
                "y el entrenamiento. El plan exige comparar las 12 combinaciones y "
                "confirmar la elección con cinco vuelos nuevos.",
                "Plan, páginas 2, 3 y 5",
            )
        name = result.best_model
        grid = predict_all_designs(result, name)
        if not has_unique_optimum(grid):
            return (
                f"El modelo con menor MAE es {name}, pero no distingue una configuración "
                "ganadora única. No es correcto anunciar un óptimo con este resultado.",
                "Resultados calculados con tus mediciones",
            )
        top = grid.iloc[0]
        return (
            f"El modelo con menor MAE ({name}) recomienda un doblez de "
            f"{top['doblez_cm']:g} cm con {int(top['clips'])} clips; "
            f"predicción: {top['estimado_m']:.2f} m. Esta predicción debe "
            "confirmarse con vuelos adicionales.",
            "Resultados calculados con tus mediciones y protocolo del plan, página 5",
        )

    # Enrutamiento por términos distintivos del propio plan.
    # La búsqueda TF-IDF queda para preguntas que no coincidan aquí.
    rules = [
        ({"presupuesto", "costo", "costos", "cuesta", "soles", "materiales"}, 15),
        ({"objetivo", "objetivos", "finalidad", "proposito", "titulo"}, 0),
        ({"hipotesis", "justificacion", "investigacion", "planteamiento"}, 1),
        ({"variables", "variable", "entradas"}, 2),
        ({"ecuacion", "formula", "algebraica", "algebraico", "coeficientes"}, 3),
        ({"48", "repeticiones"}, 5),
        ({"36", "separan", "division", "entrenamiento"}, 6),
        ({"random", "forest", "bosque", "arboles"}, 7),
        ({"constant", "modelos", "promedio"}, 8),
        ({"mae", "error", "errores"}, 9),
        ({"optimizacion", "optimizar", "maximizar"}, 11),
        ({"confirmacion", "verificacion", "comprobar", "10"}, 12),
        ({"control", "controles", "cansancio", "lanzador"}, 13),
        ({"orange", "software", "programa", "herramienta"}, 14),
        ({"jurado", "exposicion", "presentacion", "defensa"}, 16),
        ({"limitacion", "limitaciones", "desventajas", "debilidad"}, 17),
        ({"mejoras", "mejorar", "futuro"}, 18),
        ({"inventados", "simulados", "ficticios"}, 19),
    ]
    for terms, position in rules:
        if words & terms:
            record = QA[position]
            return record["answer"], f"Plan de AEROMAX-IA, {record['page']}"

    q_vec = _vectorizer.transform([question])
    scores = cosine_similarity(q_vec, _matrix)[0]
    idx = int(scores.argmax())
    if scores[idx] < 0.12:
        return (
            "No encuentro una respuesta suficientemente respaldada por el plan. "
            "Pregunta sobre variables, ecuaciones, metodología, IA, MAE, "
            "optimización, materiales o exposición. No completaré datos que "
            "el documento aún no contiene.",
            "Límite del contenido disponible",
        )
    match = QA[idx]
    return str(match["answer"]), f"Plan de AEROMAX-IA, {match['page']}"
