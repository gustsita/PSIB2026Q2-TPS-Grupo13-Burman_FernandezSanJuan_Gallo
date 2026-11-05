"""Parte 5 – Comparación entre condiciones y relación con el desempeño.

Unidad de análisis: un valor por sujeto y condición (por ejemplo, la mediana de la
potencia beta de sus épocas válidas en esa condición). Las comparaciones son DENTRO de
cada sujeto y de cada experimento (los experimentos tienen personas distintas).

Pruebas (no paramétricas, porque hay unos 10 sujetos y no se puede suponer normalidad):
- Dos condiciones pareadas (1-back vs 3-back; tarea vs pausa): Wilcoxon de rangos con signo.
- Tres o más condiciones pareadas (sesiones de un experimento): Friedman y, si da
  significativo, Wilcoxon de a pares contra la condición sin estímulo.
- Corrección por comparaciones múltiples: Holm (menos conservadora que Bonferroni).
- Tamaño del efecto: r = Z / √N para Wilcoxon; W de Kendall para Friedman.

Relación con el desempeño: potencia beta (por bloque o por sesión) vs aciertos y tiempo
de reacción. Se ajusta un modelo lineal y uno cuadrático y se comparan (AIC). Una U
invertida (coeficiente cuadrático negativo y significativo) sería compatible con la ley
de Yerkes-Dodson.

Antes de generalizar, se reproduce el análisis del artículo en los sujetos A9 y B9.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def tabla_por_condicion(tabla_epocas, valor="beta_abs", agregado="median"):
    """Pivot sujeto × condición con el valor agregado de las épocas válidas.

    TODO: filtrar valida == True, groupby(["sujeto", "condicion"])[valor].agg(agregado)
    y .unstack("condicion").
    """
    raise NotImplementedError


def wilcoxon_pareado(a, b):
    """Wilcoxon de rangos con signo entre dos condiciones pareadas.

    Devuelve dict con estadístico, p, n y el tamaño del efecto r = Z / √N.
    TODO: stats.wilcoxon(a, b, method="approx") devuelve también zstatistic.
    """
    raise NotImplementedError


def friedman(tabla):
    """Friedman sobre un pivot sujeto × condición (sin NaN). Devuelve estadístico, p y W de Kendall.

    W = χ² / (n · (k − 1)), con n sujetos y k condiciones.
    """
    tabla = tabla.dropna()
    chi2, p = stats.friedmanchisquare(*[tabla[c] for c in tabla.columns])
    n, k = tabla.shape
    return {"chi2": chi2, "p": p, "n": n, "W_kendall": chi2 / (n * (k - 1))}


def corregir_holm(pvalores):
    """p-valores corregidos por Holm."""
    return multipletests(pvalores, method="holm")[1]


def lineal_vs_cuadratico(x, y):
    """Ajusta y = a + b·x y y = a + b·x + c·x²; devuelve coeficientes y AIC de cada uno.

    TODO: np.polyfit para los coeficientes; AIC = n·ln(SSE/n) + 2k.
    """
    raise NotImplementedError
