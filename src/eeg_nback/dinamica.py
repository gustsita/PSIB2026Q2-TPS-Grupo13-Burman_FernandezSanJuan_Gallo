"""Parte 4 – Evolución temporal de la potencia beta: periodograma de Lomb.

Para cada época válida se calcula la potencia beta y se la ubica en el tiempo (centro de
la época). Como las épocas con artefactos se descartan, la serie tiene huecos: está
muestreada en forma irregular y no se le puede aplicar la FFT directamente.

El periodograma de Lomb-Scargle ajusta, para cada frecuencia, una senoidal por mínimos
cuadrados usando solo los instantes donde hay dato, así que no necesita interpolar.
Normalizado, su valor en cada frecuencia es la fracción de la varianza de la serie que
explica una senoidal de esa frecuencia (entre 0 y 1).

Qué se espera encontrar: la tarea alterna bloques de unos 43 s (22 estímulos × 2 s) con
pausas de unos 15 s, es decir un ciclo de unos 58 s (≈ 0,017 Hz). Si la potencia beta
sube durante la tarea y baja en las pausas, el periodograma debería tener un pico cerca
de esa frecuencia. La frecuencia del ciclo se mide en cada sesión a partir del inicio de
sus bloques (frecuencia_ciclo), en lugar de suponerla.

Decisiones:
- Escala logarítmica: la serie es log10 de la potencia beta. La potencia de una época
  tiene una distribución asimétrica (Parte 3, sección 5.2) y el ajuste por mínimos
  cuadrados es sensible a los valores altos: unas pocas épocas podrían dominar el
  periodograma, el mismo problema que llevó a usar la mediana en la Parte 3. En escala
  logarítmica un cambio de ×1,2 pesa igual en cualquier sujeto o canal.
- Se resta la tendencia lineal de cada sesión: una deriva lenta a lo largo de los 16 min
  (cansancio, cambio de contacto del electrodo) concentra potencia en las frecuencias más
  bajas y puede tapar el pico del ciclo, que está cerca.
- Significación por permutación: se reordenan al azar los valores de la serie sobre los
  mismos instantes (se conservan los huecos y la distribución de valores, se destruye el
  orden temporal) y se calcula cuántas veces el periodograma de la serie desordenada
  supera al real en la frecuencia del ciclo. Así no hace falta suponer que el ruido es
  gaussiano ni independiente, como pide la fórmula clásica de la probabilidad de falsa
  alarma.

Comparación: interpolar la serie a un muestreo uniforme con cubic splines y aplicar
Welch. La interpolación "inventa" valores en los huecos, y eso sesga el espectro.
"""

import numpy as np
import pandas as pd
from scipy import interpolate, signal


def serie_beta(t_centro, potencia_beta, valida):
    """Instantes (s desde el inicio de la sesión) y potencia beta de las épocas válidas."""
    t = np.asarray(t_centro)[valida]
    return t - t[0], np.asarray(potencia_beta)[valida]


def preparar(t, y, log=True, sin_tendencia=True):
    """Pasa a log10 (si log) y resta la tendencia lineal (si sin_tendencia) o la media."""
    t = np.asarray(t, dtype=float)
    y = np.log10(np.asarray(y, dtype=float)) if log else np.asarray(y, dtype=float)
    if sin_tendencia:
        return y - np.polyval(np.polyfit(t, y, 1), t)
    return y - y.mean()


def lomb(t, y, f):
    """Periodograma de Lomb normalizado para las frecuencias f (Hz).

    Resta la media de y; scipy.signal.lombscargle espera frecuencias ANGULARES
    (w = 2π f). Devuelve la fracción de la varianza explicada en cada frecuencia.
    """
    y = np.asarray(y, dtype=float)
    w = 2 * np.pi * np.asarray(f, dtype=float)
    return signal.lombscargle(np.asarray(t, dtype=float), y - y.mean(), w, normalize=True)


def spline_welch(t, y, fs_interp=0.5, nperseg=128):
    """Interpola la serie con cubic splines a fs_interp Hz y le aplica Welch.

    fs_interp = 0,5 Hz coincide con el paso de las épocas (2 s), así que en los tramos sin
    huecos la serie interpolada pasa por los mismos valores. nperseg = 128 muestras son
    256 s (unos 4 ciclos tarea-pausa), con resolución de 0,004 Hz.

    Devuelve (f, pxx, tu, yu): el espectro y la serie interpolada (para graficarla).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    tu = np.arange(t[0], t[-1], 1 / fs_interp)
    yu = interpolate.CubicSpline(t, y)(tu)
    f, pxx = signal.welch(yu - yu.mean(), fs=fs_interp, nperseg=min(nperseg, len(yu)))
    return f, pxx, tu, yu


def frecuencia_ciclo(bloques_sesion):
    """Frecuencia del ciclo tarea-pausa (Hz): inversa de la mediana entre inicios de bloque."""
    t_ini = np.sort(np.asarray(bloques_sesion["t_ini"], dtype=float))
    return 1.0 / np.median(np.diff(t_ini))


def potencia_en(f, pgram, f0, ancho=None):
    """Máximo del periodograma en f0 ± ancho (por defecto ± un paso de f)."""
    f = np.asarray(f)
    ancho = ancho if ancho is not None else (f[1] - f[0])
    return pgram[np.abs(f - f0) <= ancho].max()


def permutacion(t, y, f, f0, n_perm=500, seed=13, ancho=None):
    """p-valor de que el periodograma en f0 sea tan alto como el observado por azar.

    Desordena y sobre los mismos instantes t. Devuelve (p, nulo), con nulo = el valor en
    f0 de cada serie desordenada (para graficar la distribución).
    """
    rng = np.random.default_rng(seed)
    f = np.asarray(f)
    ancho = ancho if ancho is not None else (f[1] - f[0])
    fs_ = f[np.abs(f - f0) <= ancho]      # solo hace falta el periodograma cerca de f0
    real = lomb(t, y, fs_).max()
    nulo = np.array([lomb(t, rng.permutation(y), fs_).max() for _ in range(n_perm)])
    return (1 + np.sum(nulo >= real)) / (n_perm + 1), nulo


def analizar_sesiones(tabla_epocas, bloques, f, columna="beta", valida="usable_beta",
                      min_epocas=200, n_perm=500, log=True, sin_tendencia=True, seed=13):
    """Lomb de la serie de beta de cada sujeto, sesión y canal con suficientes épocas.

    - tabla_epocas: outputs/potencias_epocas.csv (una fila por época y canal).
    - bloques: tabla de bloques de todos los sujetos (datos.tabla_bloques con una columna
      sujeto), para medir la frecuencia del ciclo de cada sesión.

    Devuelve una fila por serie: sujeto, experimento, sesion, condicion, canal, n_epocas,
    f_ciclo, p_ciclo (fracción de varianza en f_ciclo), f_pico y p_pico (frecuencia y
    valor del máximo del periodograma), es_pico_ciclo (el máximo cae en f_ciclo) y
    p_valor (permutación).
    """
    filas = []
    claves = ["sujeto", "experimento", "sesion", "condicion", "canal"]
    for clave, g in tabla_epocas[tabla_epocas[valida]].groupby(claves, sort=False):
        if len(g) < min_epocas:
            continue
        sujeto, _, sesion, _, _ = clave
        g = g.sort_values("t_centro")
        t, y = serie_beta(g["t_centro"], g[columna], np.ones(len(g), bool))
        y = preparar(t, y, log, sin_tendencia)
        b = bloques[(bloques["sujeto"] == sujeto) & (bloques["sesion"] == sesion)]
        f0 = frecuencia_ciclo(b)
        pg = lomb(t, y, f)
        p, _ = permutacion(t, y, f, f0, n_perm, seed)
        i = np.argmax(pg)
        filas.append(dict(zip(claves, clave)) | {
            "n_epocas": len(g), "f_ciclo": f0, "p_ciclo": potencia_en(f, pg, f0),
            "f_pico": f[i], "p_pico": pg[i],
            "es_pico_ciclo": abs(f[i] - f0) <= (f[1] - f[0]), "p_valor": p})
    return pd.DataFrame(filas)