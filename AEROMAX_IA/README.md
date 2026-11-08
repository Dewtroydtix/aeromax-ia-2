# AEROMAX-IA — aplicación de aprendizaje automático

Aplicación local basada en **AEROMAX-IA: Inteligencia artificial para optimizar el vuelo de aviones de papel**, plan institucional de la I.E.P. Juan Pablo Peregrino, EXPOMATEMÁTICA 2026, categoría B.

## Qué hace

- Registrar o importar **48 distancias reales** correspondientes a **12 configuraciones × 4 vuelos**.
- Separar la **repetición 4 de cada combinación** para las **12 pruebas**, y usar los otros **36** datos para entrenar.
- Entrenar **Random Forest**, **regresión lineal** y **modelo promedio (Constant)**.
- Calcular el **error absoluto medio (MAE)** de los tres modelos en las mismas 12 pruebas.
- Predecir la distancia de las **12 configuraciones autorizadas** y señalar una recomendación solo si hay un máximo único.
- Comparar **5 vuelos nuevos** del diseño recomendado contra **5 vuelos nuevos** del diseño de referencia.
- Resolver preguntas comunes sobre el PDF mediante respuestas verificadas y consultar métricas reales ya calculadas.

**No se han cargado mediciones ficticias.** No se garantiza que Random Forest supere al promedio o a la regresión lineal.

## Requisitos

- Computadora con **Python 3.10 o posterior**.
- Conexión a internet solo para instalar paquetes la primera vez. Luego el programa funciona localmente.
- No se necesitan claves de API ni cuentas de servicios de IA.

## Ejecución

Abre una terminal dentro de la carpeta `AEROMAX_IA` y ejecuta:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

La terminal mostrará una dirección local, normalmente `http://localhost:8501`. Ábrela en el navegador.

En Windows con `py` configurado, puedes reemplazar `python` por `py`. También puedes abrir `INICIAR_WINDOWS.bat` (requiere Python instalado). En macOS/Linux, desde la terminal ejecuta `bash INICIAR_MAC_LINUX.sh`.

## Cómo registrar datos

1. Abre la pestaña **1 · Registrar vuelos**.
2. Usa la tabla editable para ingresar las distancias en **metros**, o descarga `plantilla_48_vuelos.csv`, complétala e impórtala.
3. Las únicas configuraciones permitidas son:
   - `doblez_cm`: `0`, `0.5`, `1`.
   - `clips`: `0`, `1`, `2`, `3`.
   - `repeticion`: de `1` a `4` por combinación.
4. No modifiques el diseño básico, el tipo de papel o el lanzador entre pruebas; alterna el orden de los diseños durante la toma de datos.
5. Entrena en la pestaña **2** y descarga `entrenamiento.csv` y `pruebas.csv` si deseas llevarlos a Orange Data Mining.
6. Examina las 12 predicciones en la pestaña **3** y registra los 10 vuelos adicionales en la pestaña **4**.
7. Practica la defensa del proyecto con el asistente de la pestaña **5**.

### Formato de CSV

Columnas obligatorias:

```csv
doblez_cm,clips,distancia_m
0,0,
0,0,
0,0,
0,0,
0,1,
```

En el CSV completo hay 48 filas, 4 por cada una de las 12 configuraciones. Se aceptan también `repeticion` y `vuelo_id` (presentes en la plantilla descargable). **Si no incluyes `repeticion`**, se infiere por orden de aparición dentro de cada configuración, y el cuarto registro de cada combinación es el de prueba. Se permite decimal de coma con archivo separado por punto y coma.

## Fundamentos estadísticos

- **Predictores:** `doblez_cm`, `clips`.
- **Objetivo:** `distancia_m`.
- **Modelo algebraico:** `d = a + b*x + c*n`, estimando los coeficientes con regresión lineal.
- **MAE:** media de `|real - predicción|` en los 12 vuelos reservados.
- **Criterio del plan:** comparar el MAE del Random Forest con el de Constant.
- **Selección:** se propone por defecto el modelo con menor MAE, sin tratar la selección hecha sobre un único conjunto de prueba como una prueba concluyente. El usuario puede inspeccionar cualquiera de los tres modelos.
- **Advertencia:** con 36 mediciones de entrenamiento la generalización es limitada; el experimento compara solo las 12 configuraciones del plan.

## ¿El asistente es una IA conversacional generativa?

**No**. La pestaña de preguntas utiliza búsqueda de preguntas/respuestas verificadas con TF-IDF y referencias a páginas del PDF. No usa un servicio generativo ni inventa contenido. El **aprendizaje automático real** del proyecto está en Random Forest, que aprende relaciones entre características y distancia a partir de los datos medidos.

## Estructura

```text
AEROMAX_IA/
├── app.py                    # Interfaz Streamlit
├── aeromax_core.py           # Validación, entrenamiento y cálculos
├── project_knowledge.py      # Respuestas fundamentadas en el PDF
├── PLAN_AEROMAX_IA.pdf       # Documento de referencia aportado
├── plantilla_48_vuelos.csv   # CSV vacío para registrar mediciones
├── requirements.txt
├── INICIAR_WINDOWS.bat
├── INICIAR_MAC_LINUX.sh
├── .streamlit/
│   └── config.toml
└── tests/
    └── test_core.py
```

## Tests automatizados

```bash
python -m unittest discover -s tests -v
```

Los tests fabrican datos **solo dentro de la prueba de software**; esos datos no se entregan ni muestran como resultados experimentales del proyecto.

## Seguridad

Realizar los vuelos en un lugar interior despejado, sin apuntar a personas y con supervisión docente, de acuerdo con el plan original.
