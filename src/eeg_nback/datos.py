"""Parte 0 – Carga y sincronización de los datos.

Estructura esperada (como viene de PhysioNet):

    DATA_DIR/
        Experiment_1/   A1/ ... A10/   (música)
        Experiment_2/   B1/ ... B10/   (perfume y café)
    cada sujeto:
        EEG_recording.csv     timestamps (UNIX, s), TP9, AF7, AF8, TP10, Right AUX (µV)
        tags.csv              una marca UNIX por línea: inicio y fin de cada sesión
        n_back_responses.csv  exportación de E-Prime: un estímulo por fila

Las carpetas "Excluded_ID_*" son sujetos que el artículo excluyó; no se usan.

Cómo se sincroniza todo:
- tags.csv tiene 2 marcas por sesión (inicio, fin), en el orden de las sesiones.
  Excepción: a A8 le falta el inicio de la sesión 2 (ver tabla_sesiones). Por eso los
  tags no se toman de a pares sino que se asignan a cada sesión según su hora.
- En n_back_responses.csv, la columna Start_time_unix es la hora de inicio de E-Prime
  (en segundos enteros) y las columnas StimulusNNN.OnsetTime son milisegundos desde ese
  inicio. NNN = 101, 102, ... es el número de sesión (100 es la práctica, se descarta).
  Hora UNIX de cada estímulo = Start_time_unix + OnsetTime / 1000.
- Cada sesión tiene 16 bloques de 22 estímulos (8 de 1-back y 8 de 3-back, en orden
  mezclado). Entre bloques hay una pausa: unos 15 s desde el fin de un bloque (último
  estímulo + 2 s) hasta el primer estímulo del siguiente, o sea unos 17 s entre el
  último estímulo de un bloque y el primero del siguiente (Notebook 00).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

EXPERIMENTOS = {1: "Experiment_1", 2: "Experiment_2"}

# Nombre de la sesión en E-Prime (columna Running[Block]) -> condición
CONDICIONES = {
    "NoMusic": "sin_musica",
    "RelaxSession": "relajante",
    "StressSession": "estimulante",
    "NewMusicSession": "ia",
    "NoAct": "basal",
    "PerfumeSession": "perfume",
    "CoffeeSession": "cafe",
}

_PAUSA_ENTRE_BLOQUES = 4.0   # s; dentro de un bloque los estímulos están cada ~2 s

# Si a una sesión le falta el tag de inicio (o de fin), se estima con este margen respecto
# del primer (o último) estímulo. Es la mediana de ese margen en las 66 sesiones que tienen
# sus dos tags (19 sujetos, todos menos A8). Se usa la mediana y no la media (17,4 s)
# porque hay sesiones con márgenes de hasta 90 s que la desplazan.
MARGEN_TAG_INI = 11.8   # s antes del primer estímulo (rango observado: 6,0 a 90,6 s)
MARGEN_TAG_FIN = 15.7   # s después del último estímulo (rango observado: 11,7 a 23,1 s)


def listar_sujetos(data_dir=None):
    """DataFrame con una fila por sujeto válido: experimento, sujeto, ruta."""
    data_dir = Path(data_dir) if data_dir else load_config()["DATA_DIR"]
    filas = []
    for exp, carpeta in EXPERIMENTOS.items():
        for ruta in sorted((data_dir / carpeta).glob("*")):
            if not ruta.is_dir() or ruta.name.startswith("Excluded"):
                continue
            filas.append({
                "experimento": exp,
                "sujeto": ruta.name,
                "ruta": ruta,
                "tiene_eeg": (ruta / "EEG_recording.csv").exists(),
            })
    if not filas:
        raise FileNotFoundError(
            f"No hay sujetos en {data_dir}. Bajar los datos (ver README, sección Datos)."
        )
    # A1, A2, ..., A10 en orden numérico
    df = pd.DataFrame(filas)
    df["_n"] = df["sujeto"].str[1:].astype(int)
    return df.sort_values(["experimento", "_n"]).drop(columns="_n").reset_index(drop=True)


def cargar_eeg(ruta_sujeto, canales=None):
    """EEG crudo del sujeto.

    Devuelve (t, x):
    - t: array (n,) con la hora UNIX de cada muestra, en segundos;
    - x: DataFrame (n, canales) en µV.

    Los timestamps están redondeados al milisegundo (Δt de 3 o 4 ms), así que sirven para
    ubicar la señal en el tiempo pero no para estimar fs: usar FS = 256 Hz de config.yaml.
    """
    canales = canales or load_config()["CANALES"]
    df = pd.read_csv(Path(ruta_sujeto) / "EEG_recording.csv")
    return df["timestamps"].to_numpy(), df[canales]


def cargar_tags(ruta_sujeto):
    """Marcas UNIX de tags.csv (inicio y fin de cada sesión)."""
    return np.atleast_1d(np.loadtxt(Path(ruta_sujeto) / "tags.csv"))


def cargar_nback(ruta_sujeto):
    """Un estímulo por fila, con su sesión, bloque, hora UNIX y respuesta.

    Columnas: sesion (1, 2, ...), condicion, bloque (1-16 dentro de la sesión),
    tipo ("1-back" o "3-back"), t (hora UNIX del estímulo), es_objetivo,
    acierto (bool) y rt (ms desde el inicio del estímulo; NaN si no respondió).

    El sujeto puede responder durante el estímulo (500 ms) o durante la cruz de
    fijación siguiente (1500 ms): E-Prime lo guarda en columnas distintas
    (StimulusNNN.* y FixationNNN.*). Acá se unifican.
    """
    df = pd.read_csv(Path(ruta_sujeto) / "n_back_responses.csv")
    t0 = df["Start_time_unix"].iloc[0]

    partes = []
    for nnn in range(101, 105):
        col_on = f"Stimulus{nnn}.OnsetTime"
        if col_on not in df:
            continue
        s = df[df[col_on].notna()]
        resp_estimulo = s[f"Stimulus{nnn}.RESP"].notna()
        resp_fijacion = s[f"Fixation{nnn}.RESP"].notna()
        rt = np.where(resp_estimulo, s[f"Stimulus{nnn}.RT"],
                      np.where(resp_fijacion, 500 + s[f"Fixation{nnn}.RT"], np.nan))
        partes.append(pd.DataFrame({
            "sesion": nnn - 100,
            "condicion": s["Running[Block]"].map(CONDICIONES).to_numpy(),
            "tipo": np.where(s["Instruction"].str.contains("Three|3-Back", case=False),
                             "3-back", "1-back"),
            "t": t0 + s[col_on].to_numpy() / 1000,
            "es_objetivo": (s["Correct"] == 4).to_numpy(),  # 4 = botón TARGET, 3 = NO TARGET
            "acierto": ((s[f"Stimulus{nnn}.ACC"] == 1) | (s[f"Fixation{nnn}.ACC"] == 1)).to_numpy(),
            "rt": rt,
        }))

    nb = pd.concat(partes, ignore_index=True)
    # Un bloque nuevo empieza cuando la pausa con el estímulo anterior es larga
    nuevo = nb.groupby("sesion")["t"].diff().fillna(np.inf) > _PAUSA_ENTRE_BLOQUES
    nb.insert(2, "bloque", nuevo.groupby(nb["sesion"]).cumsum().astype(int))
    return nb


def tabla_bloques(nback, dur_estimulo=2.0):
    """Un bloque por fila: sesión, condición, tipo, inicio y fin (UNIX) y desempeño."""
    g = nback.groupby(["sesion", "condicion", "bloque", "tipo"], sort=False)
    bloques = g.agg(
        t_ini=("t", "min"),
        t_fin=("t", "max"),
        n_estimulos=("t", "size"),
        aciertos=("acierto", "mean"),
        rt_mediana=("rt", "median"),
    ).reset_index()
    bloques["t_fin"] += dur_estimulo   # el último estímulo dura 500 ms + 1500 ms de fijación
    return bloques


def tabla_sesiones(ruta_sujeto, nback=None):
    """Una sesión por fila: condición, inicio y fin según tags.csv, y según E-Prime.

    Sirve para verificar la sincronización: los estímulos de cada sesión tienen que
    caer dentro del intervalo marcado por los tags.

    Los tags se asignan por hora, no de a pares: el tag de inicio de una sesión es el
    último tag entre el final de la sesión anterior y su primer estímulo; el de fin, el
    primer tag entre su último estímulo y el comienzo de la sesión siguiente.

    Si falta alguno (solo pasa en A8, sesión 2), se estima con MARGEN_TAG_INI o
    MARGEN_TAG_FIN y la columna tag_ini_estimado (o tag_fin_estimado) queda en True.
    En ese caso `dentro` se cumple por construcción y no verifica nada para esa sesión.
    """
    nback = cargar_nback(ruta_sujeto) if nback is None else nback
    tags = cargar_tags(ruta_sujeto)
    ses = nback.groupby(["sesion", "condicion"], sort=False)["t"].agg(
        primer_estimulo="min", ultimo_estimulo="max").reset_index()

    proximo = np.concatenate([ses["primer_estimulo"].iloc[1:], [np.inf]])
    usados = np.zeros(len(tags), dtype=bool)
    ini, fin, ini_est, fin_est = [], [], [], []
    libre = np.ones(len(tags), dtype=bool)   # cada tag se usa una sola vez, en orden
    for i in range(len(ses)):
        a, b = ses["primer_estimulo"].iloc[i], ses["ultimo_estimulo"].iloc[i]
        cand_ini = np.flatnonzero(libre & (tags <= a))
        libre[: cand_ini[-1] + 1 if len(cand_ini) else 0] = False
        cand_fin = np.flatnonzero(libre & (tags >= b) & (tags < proximo[i]))
        libre[: cand_fin[0] + 1 if len(cand_fin) else 0] = False
        ini_est.append(len(cand_ini) == 0)
        fin_est.append(len(cand_fin) == 0)
        ini.append(a - MARGEN_TAG_INI if ini_est[-1] else tags[cand_ini[-1]])
        fin.append(b + MARGEN_TAG_FIN if fin_est[-1] else tags[cand_fin[0]])
        usados[cand_ini[-1:]] = usados[cand_fin[:1]] = True
    if not usados.all():
        raise ValueError(f"{ruta_sujeto}: hay tags que no corresponden a ninguna sesión: {tags[~usados]}")

    ses["tag_ini"], ses["tag_fin"] = ini, fin
    ses["tag_ini_estimado"], ses["tag_fin_estimado"] = ini_est, fin_est
    ses["dentro"] = (ses["primer_estimulo"] >= ses["tag_ini"]) & (ses["ultimo_estimulo"] <= ses["tag_fin"])
    return ses


def recortar(t, x, t_ini, t_fin):
    """Tramo de la señal entre dos horas UNIX. Devuelve (t, x) recortados."""
    m = (t >= t_ini) & (t < t_fin)
    return t[m], x[m]
