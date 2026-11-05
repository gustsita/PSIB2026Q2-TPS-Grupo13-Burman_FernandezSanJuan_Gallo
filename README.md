# PSIB 2026Q2 – TPS – Grupo 13

**Ritmos cerebrales en EEG portátil durante una tarea de memoria de trabajo: carga cognitiva y estímulos cotidianos (música, café y perfume)**

Lucía Burman · Lucía Fernandez San Juan · Agustina María Gallo

El trabajo procesa el EEG de 4 canales de una vincha Muse registrado mientras los sujetos hacen una tarea n-back. Separa la señal por sesión y por bloque de tarea, la filtra (pasa banda y filtro adaptativo LMS para la interferencia de línea), la recorta en épocas y descarta las que tienen artefactos, estima la potencia de los ritmos cerebrales (con énfasis en la banda beta) y evalúa si cambia con la dificultad de la tarea y con los estímulos, y si se relaciona con el desempeño.

## Estructura

```
config.yaml               # rutas, frecuencia de muestreo, canales, bandas y frecuencia de línea
requirements.txt          # dependencias
scripts/
    descargar_datos.py    # baja de PhysioNet solo los archivos que usamos
    check_env.py          # verifica dependencias, datos y sincronización
    generar_deflexiones.py # genera la tabla de la sección 4 del Notebook 03
    generar_potencias.py  # genera las tablas de potencias (Notebook 03, Partes 4 y 5)
    prueba_ica.py         # prueba de ICA para las deflexiones (docs/DECISIONES.md)
src/eeg_nback/
    config.py             # lee config.yaml
    datos.py              # Parte 0: carga del EEG, sesiones, bloques y respuestas del n-back
    filtrado.py           # Parte 1: pasa banda Butterworth + LMS para 60 Hz
    epocas.py             # Parte 2: recorte en realizaciones y rechazo de artefactos
    espectro.py           # Parte 3: PSD de Welch (y AR como verificación), potencia por banda
    potencias.py          # Parte 3: tabla de potencias por sujeto, condición y canal
    dinamica.py           # Parte 4: periodograma de Lomb sobre la serie de potencia beta
    estadistica.py        # Parte 5: Wilcoxon / Friedman / Holm y relación beta-desempeño
notebooks/                # un notebook por parte (00 = exploración y sincronización)
data/raw/                 # datos (NO se suben a GitHub)
outputs/                  # figuras y tablas generadas (no se suben)
docs/                     # anteproyecto y DECISIONES.md (cada decisión explicada)
```

## Puesta en marcha

Requiere Python 3.10 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
```

Para Jupyter / VS Code, elegir el kernel `.venv` (o registrarlo con
`python -m ipykernel install --user --name psib-tps-g13 --display-name "PSIB TPS Grupo 13"`).

## Datos

Dataset: **Regulation of Brain Cognitive States through Auditory, Gustatory, and Olfactory Stimulation with Wearable Monitoring** (PhysioNet, v1.0.0, DOI 10.13026/jr6f-wc85).
Artículo: Fekri Azgomi H. et al., *Scientific Reports* 13, 12399 (2023). DOI 10.1038/s41598-023-37829-z.

| Experimento | Sujetos | Sesiones (en orden) |
|---|---|---|
| 1 – música | A1–A10 | sin música, relajante, estimulante, generada por IA |
| 2 – perfume y café | B1–B10 | basal, perfume, café |

Por sujeto usamos tres archivos: `EEG_recording.csv` (unos 70 MB), `tags.csv` y `n_back_responses.csv`.
Las señales de las pulseras Empatica no se usan.

1. Bajar los datos con el script (usa `curl`, continúa descargas cortadas y saltea lo que ya está):
   ```bash
   python scripts/descargar_datos.py B1 A9    # para empezar: los dos sujetos que analiza el artículo
   python scripts/descargar_datos.py          # los 20 sujetos (unos 1,4 GB)
   ```
   PhysioNet puede ser lento: conviene empezar con uno o dos sujetos y dejar el resto bajando.
2. Tiene que quedar `data/raw/Experiment_1/A1/...` y `data/raw/Experiment_2/B1/...`.
3. Correr `python scripts/check_env.py`: muestra las sesiones de cada sujeto y verifica que
   los estímulos de cada sesión caigan dentro de sus tags.

## Uso

Los notebooks importan el paquete desde `src/`:

```python
from eeg_nback import datos

sujetos = datos.listar_sujetos()              # experimento, sujeto, ruta
s = sujetos.query("sujeto == 'B1'").iloc[0]

t, eeg = datos.cargar_eeg(s.ruta)             # hora UNIX de cada muestra, DataFrame en µV
nback = datos.cargar_nback(s.ruta)            # un estímulo por fila: sesión, bloque, tipo, acierto, rt
bloques = datos.tabla_bloques(nback)          # un bloque por fila: inicio, fin, tipo, % de aciertos
sesiones = datos.tabla_sesiones(s.ruta)       # inicio y fin de cada sesión según tags.csv
```

## Plan de trabajo por módulos

Cada parte es un módulo en `src/eeg_nback/` (las funciones) y un notebook (las pruebas,
figuras y conclusiones). Los módulos traen la explicación del método. `docs/DECISIONES.md`
explica cada decisión del trabajo.

| Parte | Módulo | Estado |
|---|---|---|
| 0 | `datos.py` + `00_exploracion` | Terminada |
| 1 | `filtrado.py` + `01_filtrado` | Terminada |
| 2 | `epocas.py` + `02_epocas_artefactos` | Terminada |
| 3 | `espectro.py`, `potencias.py` + `03_ritmos` | Terminada |
| 4 | `dinamica.py` + `04_lomb` | Falta (funciones con `TODO`) |
| 5 | `estadistica.py` + `05_estadistica` | Falta (funciones con `TODO`) |

Las Partes 4 y 5 usan las tablas de `scripts/generar_potencias.py` y se pueden hacer en
paralelo. El notebook 03 necesita además `scripts/generar_deflexiones.py`.

## Flujo de trabajo en Git

1. Antes de empezar: `git switch main && git pull`.
2. Rama por tarea: `git switch -c parte2-epocas`.
3. Commits chicos y frecuentes, con mensajes que digan qué se hizo:
   `epocas: recorte en épocas de 2 s dentro de cada bloque`,
   `epocas: criterio de curtosis y z-score robusto de la varianza`.
4. `git push -u origin parte2-epocas` y abrir un Pull Request a `main`. Otra integrante
   lo revisa y lo mergea.
5. Cada integrante commitea con su propio usuario de GitHub: la cátedra audita la
   distribución de aportes entre integrantes.
