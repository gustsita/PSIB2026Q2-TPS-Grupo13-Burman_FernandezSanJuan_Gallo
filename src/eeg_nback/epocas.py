"""Parte 2 – Recorte en realizaciones (épocas) y rechazo de artefactos.

Idea: dentro de un tramo "estable" (un bloque de 1-back, de 3-back o una pausa) el EEG
se puede tratar como un proceso estocástico estacionario en sentido amplio. Cada época
de 2-4 s es una realización de ese proceso. Las épocas contaminadas se descartan antes
de estimar el espectro, porque un solo parpadeo (cientos de µV) domina la PSD.

Criterios de rechazo (por canal y por época):
1. Amplitud pico a pico mayor que un umbral (parpadeos, movimiento, saturación).
2. Varianza muy distinta del resto de las épocas del mismo canal. Se usa un z-score
   robusto: (v − mediana) / (1,4826 · MAD), para que los artefactos no "inflen" el
   desvío con el que se los compara.
3. Curtosis alta: un parpadeo es un evento aislado de gran amplitud, que hace que la
   distribución de la época tenga colas pesadas (curtosis >> 3).
4. Señal casi plana (desvío muy chico): electrodo sin contacto.
5. Muestras saturadas (±1000 µV), marcadas sobre la señal CRUDA con
   filtrado.mascara_saturacion antes de filtrar.

Lo que ya se vio en B1 (para tenerlo en cuenta al elegir umbrales): AF7 tiene una PSD
mucho mayor que los otros canales en todas las bandas y cerca del 0,7 % de sus muestras
saturadas en las sesiones basal y perfume; AF8, en cambio, tiene una potencia en alfa
unas 10 veces menor que TP9 y TP10. Puede ser mal contacto: hay que confirmarlo en
todos los sujetos antes de decidir si se usa cada canal.

Los umbrales se eligen mirando las distribuciones de cada criterio (Notebook 02), no
a ojo: se justifican con percentiles y se informa cuántas épocas descarta cada uno.
"""

import numpy as np
import pandas as pd
from scipy import stats


def recortar_epocas(t, x, t_ini, t_fin, fs, dur=2.0, solape=0.0):
    """Recorta el tramo [t_ini, t_fin) en épocas de `dur` segundos.

    Devuelve (epocas, t_centro):
    - epocas: array (n_epocas, n_muestras, n_canales);
    - t_centro: hora UNIX del centro de cada época (la usa la Parte 4).

    TODO:
    - quedarse con las muestras del tramo (datos.recortar);
    - n = int(dur * fs) muestras por época, paso = int(n * (1 - solape));
    - armar las épocas con un loop o con np.lib.stride_tricks.sliding_window_view;
    - descartar la última si queda incompleta.
    """
    raise NotImplementedError


def criterios(epocas):
    """DataFrame (n_epocas × n_canales filas) con los criterios de cada época y canal.

    Columnas: epoca, canal, pico_a_pico, varianza, curtosis, desvio.
    TODO: np.ptp, np.var, stats.kurtosis(fisher=False) a lo largo del eje de muestras.
    """
    raise NotImplementedError


def zscore_robusto(v):
    """(v − mediana) / (1,4826 · MAD). Con datos normales equivale a un z-score común."""
    v = np.asarray(v, dtype=float)
    mad = stats.median_abs_deviation(v, nan_policy="omit")
    return (v - np.nanmedian(v)) / (1.4826 * mad)


def marcar_artefactos(tabla, umbral_pp, umbral_z=3.5, umbral_curtosis=5.0, desvio_min=1.0):
    """Agrega a la tabla de `criterios` una columna booleana `valida` y el motivo de rechazo.

    La tabla puede traer una columna `saturada` (la época tenía muestras saturadas en la
    señal cruda); si la trae, también es motivo de rechazo.

    TODO:
    - z = zscore_robusto de la varianza, calculado por canal (groupby("canal"));
    - motivo = el primer criterio que falla ("saturada", "pico_a_pico", "varianza",
      "curtosis", "plana");
    - valida = motivo vacío.
    """
    raise NotImplementedError


def resumen_calidad(tabla):
    """Porcentaje de épocas válidas por canal (y por condición, si la tabla la tiene)."""
    claves = [c for c in ("condicion", "canal") if c in tabla]
    return (tabla.groupby(claves)["valida"].mean() * 100).rename("pct_validas").reset_index()
