"""Parte 5 – Comparación entre condiciones y relación con el desempeño.

Unidad de análisis: un valor por sujeto y condición, la mediana de la potencia de sus
épocas válidas en esa condición (Parte 3). Las comparaciones son DENTRO de cada sujeto y
de cada experimento (los experimentos tienen personas distintas).

Pruebas (no paramétricas, porque hay unos 10 sujetos y no se puede suponer normalidad):
- Dos condiciones pareadas (1-back vs 3-back; tarea vs pausa): Wilcoxon de rangos con signo.
- Tres o más condiciones pareadas (sesiones de un experimento): Friedman y Wilcoxon de a
  pares contra la condición sin estímulo.
- Corrección por comparaciones múltiples: Holm (menos conservadora que Bonferroni). Se
  aplica dentro de cada familia de pruebas (por ejemplo, los 4 canales de una misma
  comparación), no a todo el trabajo junto.
- Tamaño del efecto: r = Z / √N para Wilcoxon; W de Kendall para Friedman.

Qué épocas y canales entran: los mismos que en la tabla de potencias (potencias.py).
beta usa `usable_beta` y la regla de canales para beta; las bandas lentas y theta/beta
usan `usable` y su regla, en la práctica solo los frontales.

Relación con el desempeño: potencia beta por bloque contra aciertos y tiempo de reacción
del bloque. Se ajusta un modelo lineal y uno cuadrático y se comparan con el AIC. Una U
invertida (coeficiente cuadrático negativo y significativo) sería compatible con la ley de
Yerkes-Dodson.

Antes de generalizar, se reproduce el análisis del artículo en los sujetos A9 y B9.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from . import potencias

MIN_EPOCAS = 20   # épocas válidas mínimas para que una celda sujeto × condición cuente


def agregar_fase(tabla_epocas):
    """Agrega la columna fase: "tarea" (1-back o 3-back) o "pausa"."""
    return tabla_epocas.assign(fase=np.where(tabla_epocas["tipo"] == "pausa", "pausa", "tarea"))


def tabla_por_condicion(tabla_epocas, canal, por="tipo", columna="beta", valida="usable_beta",
                        agregado="median", min_epocas=MIN_EPOCAS, regla=None):
    """Pivot sujeto × condición con el valor agregado de las épocas válidas de un canal.

    - por: columna que define la condición ("tipo", "fase", "condicion", ...).
    - valida: "usable_beta" para beta, "usable" para las bandas lentas y los cocientes.
      Según cuál sea, se aplica la regla de canales de beta o la de las bandas lentas
      (o la que se indique en regla: "beta" o "lentas").
    - Las celdas con menos de min_epocas épocas quedan en NaN.
    """
    regla = regla or ("beta" if valida == "usable_beta" else "lentas")
    regla = potencias.cobertura(tabla_epocas)[regla]
    t = tabla_epocas[(tabla_epocas["canal"] == canal) & tabla_epocas[valida]]
    t = t[[regla.get((s, canal), False) for s in t["sujeto"]]]
    g = t.groupby(["sujeto", por])[columna]
    valor = g.agg(agregado).where(g.size() >= min_epocas)
    return valor.unstack(por)


def wilcoxon_pareado(a, b):
    """Wilcoxon de rangos con signo entre dos condiciones pareadas (b respecto de a).

    a y b son Series con el mismo índice (sujeto); se usan solo los sujetos con las dos.
    Devuelve dict con n, la mediana del cociente b / a entre sujetos, cuántos sujetos
    tienen b > a, el estadístico, p y r = Z / √N.

    p sale de la distribución exacta (con n chico es más precisa que la aproximación
    normal); Z sale de la aproximación normal y se usa solo para r. El signo de r es el
    de la mediana de las diferencias: positivo si b tiende a ser mayor que a.
    """
    d = pd.concat({"a": a, "b": b}, axis=1).dropna()
    n = len(d)
    if n < 5:
        return {"n": n, "cociente": np.nan, "n_sube": np.nan, "W": np.nan, "p": np.nan, "r": np.nan}
    dif = d["b"] - d["a"]
    res = stats.wilcoxon(d["b"], d["a"])
    z = abs(stats.wilcoxon(d["b"], d["a"], method="approx").zstatistic)
    return {"n": n, "cociente": (d["b"] / d["a"]).median(), "n_sube": int((dif > 0).sum()),
            "W": res.statistic, "p": res.pvalue, "r": np.sign(dif.median()) * z / np.sqrt(n)}


def friedman(tabla):
    """Friedman sobre un pivot sujeto × condición (sin NaN). Devuelve estadístico, p y W de Kendall.

    W = χ² / (n · (k − 1)), con n sujetos y k condiciones.
    """
    tabla = tabla.dropna()
    chi2, p = stats.friedmanchisquare(*[tabla[c] for c in tabla.columns])
    n, k = tabla.shape
    return {"chi2": chi2, "p": p, "n": n, "W_kendall": chi2 / (n * (k - 1))}


def corregir_holm(pvalores):
    """p-valores corregidos por Holm (los NaN quedan en NaN)."""
    p = np.asarray(pvalores, dtype=float)
    salida = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        salida[ok] = multipletests(p[ok], method="holm")[1]
    return salida


def comparar_canales(tabla_epocas, canales, a, b, por="tipo", **kw):
    """Wilcoxon de b contra a en cada canal, con Holm sobre los canales.

    kw se pasa a tabla_por_condicion (columna, valida, min_epocas).
    """
    filas = []
    for canal in canales:
        piv = tabla_por_condicion(tabla_epocas, canal, por=por, **kw)
        if a in piv and b in piv:
            filas.append({"canal": canal} | wilcoxon_pareado(piv[a], piv[b]))
    res = pd.DataFrame(filas)
    res["p_holm"] = corregir_holm(res["p"])
    return res


def comparar_sesiones(tabla_epocas, canales, referencia, **kw):
    """Friedman entre las sesiones de un experimento y Wilcoxon de cada una contra la referencia.

    Usa solo las épocas de tarea. Devuelve (friedman, pares):
    - friedman: una fila por canal, con Holm sobre los canales;
    - pares: una fila por canal y sesión, con Holm sobre las sesiones de cada canal.
    Friedman necesita sujetos con todas las sesiones; Wilcoxon usa los que tienen las dos.
    """
    tarea = tabla_epocas[tabla_epocas["tipo"] != "pausa"]
    fr, pares = [], []
    for canal in canales:
        piv = tabla_por_condicion(tarea, canal, por="condicion", **kw)
        fr.append({"canal": canal} | friedman(piv))
        filas = [{"canal": canal, "condicion": c} | wilcoxon_pareado(piv[referencia], piv[c])
                 for c in piv.columns if c != referencia]
        filas = pd.DataFrame(filas)
        filas["p_holm"] = corregir_holm(filas["p"])
        pares.append(filas)
    fr = pd.DataFrame(fr)
    fr["p_holm"] = corregir_holm(fr["p"])
    return fr, pd.concat(pares, ignore_index=True)


def beta_por_bloque(tabla_epocas, bloques, columna="beta", valida="usable_beta", min_epocas=10):
    """Una fila por sujeto, sesión, bloque y canal: mediana de beta y desempeño del bloque.

    - bloques: datos.tabla_bloques de todos los sujetos, con una columna sujeto.
    - min_epocas: un bloque tiene 22 épocas; se pide al menos 10 válidas.
    Agrega log_beta (log10 de la mediana) y sus z-scores dentro de cada sujeto, canal y
    tipo de bloque (ver centrar).
    """
    regla = potencias.cobertura(tabla_epocas)["beta" if valida == "usable_beta" else "lentas"]
    t = tabla_epocas[tabla_epocas[valida] & (tabla_epocas["tipo"] != "pausa")]
    t = t[[regla.get(k, False) for k in zip(t["sujeto"], t["canal"])]]
    claves = ["sujeto", "experimento", "sesion", "condicion", "bloque", "tipo", "canal"]
    g = t.groupby(claves)[columna].agg(beta="median", n_epocas="size").reset_index()
    g = g[g["n_epocas"] >= min_epocas]
    g = g.merge(bloques[["sujeto", "sesion", "bloque", "aciertos", "rt_mediana"]],
                on=["sujeto", "sesion", "bloque"])
    g["log_beta"] = np.log10(g["beta"])
    return g


def centrar(df, columnas, grupos):
    """z-score de cada columna dentro de cada grupo.

    Sirve para juntar bloques de distintos sujetos: el nivel de beta depende del contacto
    de cada electrodo (Parte 3) y el desempeño depende del sujeto y de la dificultad, así
    que se compara cada bloque con los demás del mismo sujeto, canal y tipo.
    """
    df = df.copy()
    for c in columnas:
        g = df.groupby(grupos)[c]
        df[f"z_{c}"] = (df[c] - g.transform("mean")) / g.transform("std")
    return df


def lineal_vs_cuadratico(x, y):
    """Ajusta y = a + b·x y y = a + b·x + c·x² por mínimos cuadrados.

    Devuelve dict con los coeficientes, el p-valor de c (prueba t del modelo cuadrático),
    el AIC de cada modelo y delta_aic = AIC cuadrático − AIC lineal (negativo: el
    cuadrático explica lo suficiente como para justificar el parámetro extra).

    AIC = n·ln(SSE/n) + 2k, con k = cantidad de coeficientes (2 o 3).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    lin = sm.OLS(y, sm.add_constant(x)).fit()
    cua = sm.OLS(y, np.column_stack([np.ones(n), x, x ** 2])).fit()

    def aic(m, k):
        return n * np.log(m.ssr / n) + 2 * k

    return {"n": n, "b_lineal": lin.params[1], "p_lineal": lin.pvalues[1],
            "a": cua.params[0], "b": cua.params[1], "c": cua.params[2], "p_c": cua.pvalues[2],
            "aic_lineal": aic(lin, 2), "aic_cuadratico": aic(cua, 3),
            "delta_aic": aic(cua, 3) - aic(lin, 2)}


def coeficientes_por_sujeto(df, x, y, min_bloques=20):
    """Ajuste cuadrático de y contra x en cada sujeto. Devuelve una fila por sujeto con b y c.

    Complementa el ajuste con todos los bloques juntos: si la U invertida es real, el
    coeficiente c tiene que ser negativo en la mayoría de los sujetos, y eso se prueba con
    Wilcoxon contra 0.
    """
    filas = []
    for sujeto, g in df.groupby("sujeto"):
        g = g[[x, y]].dropna()
        if len(g) < min_bloques:
            continue
        c, b, _ = np.polyfit(g[x], g[y], 2)
        filas.append({"sujeto": sujeto, "n_bloques": len(g), "b": b, "c": c})
    return pd.DataFrame(filas)
