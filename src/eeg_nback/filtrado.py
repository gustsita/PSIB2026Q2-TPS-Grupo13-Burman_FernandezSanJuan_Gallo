"""Parte 1 – Filtrado: filtro adaptativo LMS para la línea de 60 Hz + pasa banda Butterworth.

Por qué hace falta el LMS (medido en el sujeto B1, sesión basal, PSD de Welch):
la interferencia de la red eléctrica (60 Hz en EE.UU.) es muy fuerte en algunos canales.
En TP9 la PSD en 60 Hz es unas 16 veces la de 10 Hz y en AF7 unas 24 veces. El pasa
banda de 1-40 Hz la atenúa (≈ 19 dB en 60 Hz en una pasada), pero no alcanza: después
del pasa banda, el pico de 60 Hz en TP9 sigue siendo unas 18 veces su entorno
(potencia_relativa_en). Hay que confirmarlo en todos los sujetos (Notebook 01).

Orden de las etapas:
1. Saturación: la Muse satura en ±1000 µV. Esas muestras se marcan ANTES de filtrar
   (mascara_saturacion), porque el filtrado "desparrama" el escalón a las muestras
   vecinas y después ya no se distingue. La Parte 2 descarta las épocas que las contienen.
2. LMS sobre la señal cruda (sin la media): la interferencia todavía está entera y el
   filtro adaptativo la estima y la resta. Si se filtrara primero con el pasa banda, el
   LMS no tendría casi nada que cancelar y no se podría evaluar.
3. Pasa banda Butterworth 1-40 Hz con filtfilt:
   - 1 Hz: saca la deriva lenta (sudor, movimiento del electrodo). Por eso la banda delta
     se toma desde 1 Hz (config.yaml), igual que el filtrado del artículo (1-50 Hz).
   - 40 Hz: deja margen sobre beta (hasta 30 Hz) y atenúa EMG y lo que quede de 60 Hz.
   - filtfilt (fase cero): filtra ida y vuelta, así no hay retardo de fase que corra los
     eventos respecto de los bloques; el orden efectivo se duplica.
   - Se usa la forma SOS (secciones de segundo orden) por estabilidad numérica.

Nota: los timestamps de la Muse están redondeados al milisegundo (los Δt valen 3 o 4 ms),
así que fs no se estima con diff(t): se usa FS = 256 Hz de config.yaml (verificado:
n_muestras / duración = 256,00 Hz en B1).

Filtro LMS como cancelador adaptativo de ruido (ANC), con referencia sintética:
- Referencia: dos senoidales en cuadratura a la frecuencia de línea,
      r1[n] = cos(2π·60·n/fs),  r2[n] = sin(2π·60·n/fs).
  Con dos pesos w1, w2 se puede representar una senoidal de 60 Hz de cualquier amplitud
  y fase: y[n] = w1·r1[n] + w2·r2[n] es la estimación de la interferencia.
- Error = señal limpia: e[n] = x[n] − y[n].
- Actualización (descenso por gradiente estocástico del error cuadrático):
      w ← w + 2μ · e[n] · r[n]
- μ chico: converge lento pero el "notch" que forma es angosto (no come EEG vecino).
  μ grande: sigue rápido los cambios de amplitud, pero el notch se ensancha y el error en
  régimen aumenta. Se elige μ mirando la curva de convergencia de los pesos y el espectro
  resultante alrededor de 60 Hz.
"""

import numpy as np
from scipy import signal


def mascara_saturacion(x, limite=999.0):
    """Booleano (mismas dimensiones que x): True donde la muestra está saturada."""
    return np.abs(np.asarray(x)) >= limite


def disenar_pasabanda(fs, f_bajo=1.0, f_alto=40.0, orden=4):
    """Butterworth pasa banda en forma SOS."""
    return signal.butter(orden, [f_bajo, f_alto], btype="bandpass", fs=fs, output="sos")


def respuesta_frecuencia(sos, fs, n=4096):
    """(f, |H| en dB) del filtro. Con filtfilt la atenuación efectiva se duplica en dB."""
    f, h = signal.sosfreqz(sos, worN=n, fs=fs)
    return f, 20 * np.log10(np.maximum(np.abs(h), 1e-12))


def aplicar_pasabanda(x, sos):
    """Filtrado de fase cero a lo largo del tiempo (eje 0: muestras × canales)."""
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=float), axis=0)


def lms_linea(x, fs, f_linea=60.0, mu=1e-3):
    """Cancela la interferencia de línea de UN canal con un LMS de dos pesos.

    Devuelve (e, w): la señal limpia (error) y la historia de los pesos (n × 2),
    que sirve para ver la convergencia.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = np.arange(len(x))
    r1 = np.cos(2 * np.pi * f_linea * n / fs)
    r2 = np.sin(2 * np.pi * f_linea * n / fs)

    e = np.empty_like(x)
    w = np.empty((len(x), 2))
    w1 = w2 = 0.0
    # Loop con escalares de Python (más rápido que operar con arrays de 2 elementos)
    for k, (xk, a, b) in enumerate(zip(x.tolist(), r1.tolist(), r2.tolist())):
        ek = xk - (w1 * a + w2 * b)     # e[k] = x[k] − y[k]
        w1 += 2 * mu * ek * a           # w ← w + 2μ·e[k]·r[k]
        w2 += 2 * mu * ek * b
        e[k] = ek
        w[k] = w1, w2
    return e, w


def filtrar(x, fs, sos, f_linea=60.0, mu=1e-3):
    """Cadena completa sobre un tramo crudo (muestras × canales): LMS en cada canal y
    después el pasa banda. Devuelve un array del mismo tamaño que x."""
    x = np.asarray(x, dtype=float)
    limpia = np.column_stack([lms_linea(x[:, j], fs, f_linea, mu)[0] for j in range(x.shape[1])])
    return aplicar_pasabanda(limpia, sos)


def potencia_relativa_en(x, fs, f0, ancho=1.0, vecinos=(3.0, 6.0)):
    """Cociente entre la PSD en f0 ± ancho y la PSD en las bandas vecinas.

    Mide cuánto sobresale un pico (por ejemplo 60 Hz) sobre el fondo: ≈ 1 si no hay pico.
    Vecinos: de f0 − 6 a f0 − 3 Hz y de f0 + 3 a f0 + 6 Hz.
    """
    f, p = signal.welch(np.asarray(x) - np.mean(x), fs=fs, nperseg=int(4 * fs))
    banda = lambda a, b: p[(f >= a) & (f < b)].mean()  # noqa: E731
    pico = banda(f0 - ancho, f0 + ancho)
    fondo = (banda(f0 - vecinos[1], f0 - vecinos[0]) + banda(f0 + vecinos[0], f0 + vecinos[1])) / 2
    return pico / fondo
