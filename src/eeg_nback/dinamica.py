"""Parte 4 – Evolución temporal de la potencia beta: periodograma de Lomb.

Para cada época válida se calcula la potencia beta y se la ubica en el tiempo (centro de
la época). Como las épocas con artefactos se descartan, la serie tiene huecos: está
muestreada en forma irregular y no se le puede aplicar la FFT directamente.

El periodograma de Lomb-Scargle ajusta, para cada frecuencia, una senoidal por mínimos
cuadrados usando solo los instantes donde hay dato, así que no necesita interpolar.

Qué se espera encontrar: la tarea alterna bloques de unos 43 s (22 estímulos × 2 s) con
pausas de unos 15 s, es decir un ciclo de unos 58 s (≈ 0,017 Hz). Si la potencia beta
sube durante la tarea y baja en las pausas, el periodograma debería tener un pico cerca
de esa frecuencia.

Comparación: interpolar la serie a un muestreo uniforme con cubic splines y aplicar
Welch. La interpolación "inventa" valores en los huecos, y eso sesga el espectro.
"""

import numpy as np
from scipy import interpolate, signal


def serie_beta(t_centro, potencia_beta, valida):
    """Instantes (s desde el inicio de la sesión) y potencia beta de las épocas válidas."""
    t = np.asarray(t_centro)[valida]
    return t - t[0], np.asarray(potencia_beta)[valida]


def lomb(t, y, f):
    """Periodograma de Lomb normalizado para las frecuencias f (Hz).

    TODO:
    - restar la media de y;
    - scipy.signal.lombscargle espera frecuencias ANGULARES: w = 2π f;
    - pgram = signal.lombscargle(t, y, w, normalize=True).
    """
    raise NotImplementedError


def spline_welch(t, y, fs_interp=1.0):
    """Interpola la serie con cubic splines a fs_interp Hz y le aplica Welch.

    TODO:
    - tu = np.arange(t[0], t[-1], 1 / fs_interp)
    - yu = interpolate.CubicSpline(t, y)(tu)
    - signal.welch(yu - yu.mean(), fs=fs_interp, nperseg=...)
    """
    raise NotImplementedError
