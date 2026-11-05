"""Parte 3 – Ritmos cerebrales: densidad espectral de potencia y potencia por banda.

Método principal: Welch. Divide la señal en segmentos con ventana (Hann), calcula el
periodograma de cada uno y los promedia. Promediar reduce la varianza del estimador a
costa de resolución en frecuencia (Δf = fs / nperseg).

Largo del segmento: 2 s (nperseg = 2·fs), o sea la época entera. La PSD se calcula por
época y el promedio de Welch se hace entre las épocas válidas de cada condición. Con
respecto a segmentos de 1 s: Δf = 0,5 Hz en lugar de 1 Hz, menos fuga entre bandas
vecinas y menos sesgo en delta. Medido en el Notebook 03, sección 2.

Verificación: modelo AR (Yule-Walker / Levinson-Durbin). Modela la señal como la salida
de un filtro todo-polos excitado por ruido blanco; la PSD queda
    S(f) = σ² / |1 − Σ a_k e^{−j2πfk/fs}|².
El orden se elige con AIC/BIC. Se usa solo en algunos sujetos, para mostrar que las
potencias por banda no dependen del estimador.

Potencia en una banda = integral de la PSD entre sus límites (regla del trapecio).
Potencia relativa = potencia de la banda / potencia total entre 1 y 30 Hz (el pasa banda
empieza en 1 Hz, ver filtrado.py).
"""

import numpy as np
from scipy import integrate, signal


def psd_welch(x, fs, nperseg=None, solape=0.5):
    """PSD de Welch a lo largo del último eje. Devuelve (f, pxx) con pxx en µV²/Hz."""
    nperseg = nperseg or int(2 * fs)
    return signal.welch(x, fs=fs, window="hann", nperseg=nperseg,
                        noverlap=int(nperseg * solape), axis=-1)


def potencia_banda(f, pxx, banda):
    """Potencia absoluta (µV²) en `banda` = (f_min, f_max), integrando la PSD.

    Integra a lo largo del último eje, así que pxx puede ser un espectro o un array
    (épocas × canales × frecuencias).

    La máscara incluye los dos bordes. Con f < f_max el trapecio terminaría un bin antes
    del límite: con Δf = 1 Hz, delta perdería el 37 % de su potencia y theta el 26 %
    (medido con ruido blanco, Notebook 03). Incluyéndolos, las bandas contiguas se tocan
    sin huecos ni solapamiento y su suma da la potencia total.
    """
    f_min, f_max = banda
    m = (f >= f_min) & (f <= f_max)
    if m.sum() < 2:
        raise ValueError(f"La banda {banda} tiene menos de 2 frecuencias con Δf = {f[1] - f[0]} Hz.")
    return integrate.trapezoid(pxx[..., m], f[m], axis=-1)


def potencias(f, pxx, bandas, rango_total=(1, 30)):
    """Potencia de cada banda y cocientes, integrando a lo largo del último eje.

    Devuelve un dict con <banda>_abs (µV²), <banda>_rel (fracción de la potencia en
    rango_total) y theta_beta. Cada valor tiene la forma de pxx sin el eje de frecuencias.

    Cuidado con qué épocas se usan: <banda>_rel y theta_beta incluyen las frecuencias
    bajas, así que solo valen en épocas con `usable` (criterios sobre 1-40 Hz). En una
    época que solo tiene `usable_beta`, una deflexión lenta infla el denominador: en TP9
    de A1, beta_rel pasa de 0,41 a 0,03 (Notebook 03, sección 3). Ahí usar beta_abs.
    """
    total = potencia_banda(f, pxx, rango_total)
    salida = {}
    for nombre, banda in bandas.items():
        salida[f"{nombre}_abs"] = potencia_banda(f, pxx, banda)
        salida[f"{nombre}_rel"] = salida[f"{nombre}_abs"] / total
    salida["theta_beta"] = salida["theta_abs"] / salida["beta_abs"]
    return salida


def psd_ar(x, fs, orden, nfft=512):
    """PSD de un modelo AR de orden `orden` ajustado por Yule-Walker.

    TODO:
    - from statsmodels.regression.linear_model import yule_walker
      a, sigma = yule_walker(x, order=orden, method="mle")
    - f = np.linspace(0, fs / 2, nfft); z = e^{−j2πf/fs}
    - S(f) = sigma² / |1 − Σ a_k z^k|²  (escalar para que la unidad sea µV²/Hz: dividir por fs)
    """
    raise NotImplementedError


def elegir_orden_ar(x, ordenes=range(2, 31)):
    """AIC y BIC para cada orden candidato. Devuelve un DataFrame orden, aic, bic.

    TODO: from statsmodels.tsa.ar_model import AutoReg; AutoReg(x, lags=p).fit().aic / .bic
    """
    raise NotImplementedError
