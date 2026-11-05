"""Parte 3 – Tabla de potencias por sujeto, condición y canal.

Une lo anterior: filtra cada sujeto, recorta las épocas, aplica el rechazo de la Parte 2
y calcula la potencia de cada banda. La usan las Partes 4 y 5.

Qué épocas y canales usa cada medida (Parte 2 y Notebook 03, sección 3):
- beta_abs: épocas `usable_beta`, en los canales que cumplen la regla para beta.
- Todo lo demás (delta, theta, alfa, potencias relativas y theta/beta): épocas `usable`,
  en los canales que cumplen la regla para esas bandas. La potencia relativa y theta/beta
  incluyen las frecuencias bajas, así que no sirven en una época con deflexión.

Regla de canales (Parte 2): un canal se usa en un sujeto si conserva al menos el 50 % de
sus épocas de tarea en cada sesión.

Valor de cada condición: la mediana entre las épocas válidas. Es la medida principal porque
unas pocas épocas altas no la mueven (Notebook 03, sección 5.3). Los cocientes se calculan
por época y después se toma la mediana. Como control se guarda también la media: en las
potencias es el promedio de Welch, y en los cocientes es el cociente de las medias.
"""

import numpy as np
import pandas as pd

from . import epocas, espectro, filtrado

UMBRALES = dict(umbral_pp=150.0, umbral_z=3.5, umbral_curtosis=5.0, desvio_min=1.0)
UMBRAL_FRAC = 0.25
MIN_USABLES = 0.5
RANGO_TOTAL = (1, 30)
BANDAS_LENTAS = ("delta", "theta", "alfa")


def tabla_epocas(ruta, fs, canales, bandas):
    """Una fila por época y canal, con su potencia y las marcas de la Parte 2.

    Columnas: sesion, condicion, bloque, tipo, t_centro, canal, usable, usable_beta,
    una columna por banda (µV²) y total (potencia de 1 a 30 Hz). `usable` y `usable_beta`
    ya incluyen la exclusión de canales por picos angostos.
    """
    ep, info, canales_ses = epocas.epocas_sujeto(ruta, fs, canales, filtrado.disenar_pasabanda(fs))
    ep_beta, _, _ = epocas.epocas_sujeto(ruta, fs, canales, filtrado.disenar_pasabanda(fs, 13, 40))
    usar = epocas.marcar_canales(canales_ses, UMBRAL_FRAC).set_index(["sesion", "canal"])["usar"]

    f, pxx = espectro.psd_welch(np.moveaxis(ep, 1, 2), fs)
    potencia = {b: espectro.potencia_banda(f, pxx, r) for b, r in bandas.items()}
    potencia["total"] = espectro.potencia_banda(f, pxx, RANGO_TOTAL)

    def marcas(e):
        return epocas.marcar_artefactos(epocas.tabla_criterios(e, info, canales), **UMBRALES) \
            .pivot(index="epoca", columns="canal", values="valida")

    amplia, beta = marcas(ep), marcas(ep_beta)
    filas = []
    for j, canal in enumerate(canales):
        canal_ok = usar.reindex(pd.MultiIndex.from_arrays(
            [info["sesion"], [canal] * len(info)])).to_numpy(dtype=bool)
        filas.append(info[["sesion", "condicion", "bloque", "tipo", "t_centro"]].assign(
            canal=canal,
            usable=amplia[canal].to_numpy() & canal_ok,
            usable_beta=beta[canal].to_numpy() & canal_ok,
            **{nombre: valor[:, j] for nombre, valor in potencia.items()}))
    return pd.concat(filas, ignore_index=True)


def cobertura(tabla):
    """Canales que cumplen la regla del 50 %, para cada sujeto.

    Devuelve un DataFrame con índice (sujeto, canal) y columnas booleanas lentas y beta.
    """
    tarea = tabla[tabla["tipo"] != "pausa"]
    fraccion = tarea.groupby(["sujeto", "canal", "sesion"])[["usable", "usable_beta"]].mean()
    minimo = fraccion.groupby(["sujeto", "canal"]).min() >= MIN_USABLES
    return minimo.rename(columns={"usable": "lentas", "usable_beta": "beta"})


def tabla_condiciones(tabla, bandas):
    """Una fila por sujeto, condición, tipo, canal y medida, en los canales que cumplen la regla.

    Columnas: sujeto, experimento, condicion, tipo, canal, medida, valor (mediana entre
    épocas, la medida principal), media (control) y n_epocas.
    Medidas: <banda>_abs, <banda>_rel y theta_beta, con las épocas que indica el módulo.
    """
    claves = ["sujeto", "experimento", "condicion", "tipo", "canal"]
    t = tabla.join(cobertura(tabla).add_prefix("regla_"), on=["sujeto", "canal"])

    beta = t[t["usable_beta"] & t["regla_beta"]].groupby(claves)["beta"]
    salida = [beta.agg(valor="median", media="mean", n_epocas="size").reset_index()
              .assign(medida="beta_abs")]

    x = t[t["usable"] & t["regla_lentas"]]
    por_epoca = {f"{b}_abs": x[b] for b in BANDAS_LENTAS}
    por_epoca |= {f"{b}_rel": x[b] / x["total"] for b in bandas}
    por_epoca["theta_beta"] = x["theta"] / x["beta"]

    grupos = x.groupby(claves)
    medias = grupos[[*bandas, "total"]].mean()
    control = {f"{b}_abs": medias[b] for b in BANDAS_LENTAS}
    control |= {f"{b}_rel": medias[b] / medias["total"] for b in bandas}
    control["theta_beta"] = medias["theta"] / medias["beta"]

    n = grupos.size()
    for medida, valores in por_epoca.items():
        mediana = valores.groupby([x[c] for c in claves]).median()
        salida.append(pd.DataFrame({"valor": mediana, "media": control[medida], "n_epocas": n})
                      .reset_index().assign(medida=medida))
    return pd.concat(salida, ignore_index=True)
