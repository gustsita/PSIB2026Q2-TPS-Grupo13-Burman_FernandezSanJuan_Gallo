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

Promedio: la potencia de una condición es la media de las épocas válidas, que es el
promedio de Welch (la integral es lineal, así que da lo mismo promediar la PSD antes o
la potencia después). Los cocientes se calculan después de promediar: cociente de medias,
no media de cocientes.
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

    Columnas: sujeto, experimento, condicion, tipo, canal, medida, valor (media entre
    épocas), mediana (solo en las potencias absolutas) y n_epocas.
    Medidas: <banda>_abs, <banda>_rel y theta_beta, con las épocas que indica el módulo.
    """
    cob = cobertura(tabla)
    claves = ["sujeto", "experimento", "condicion", "tipo", "canal"]
    t = tabla.join(cob.add_prefix("regla_"), on=["sujeto", "canal"])

    beta = t[t["usable_beta"] & t["regla_beta"]].groupby(claves)
    salida = [beta.agg(valor=("beta", "mean"), mediana=("beta", "median"), n_epocas=("beta", "size"))
              .reset_index().assign(medida="beta_abs")]

    lentas = t[t["usable"] & t["regla_lentas"]].groupby(claves)
    medias = lentas[[*bandas, "total"]].mean()
    medianas = lentas[[*bandas]].median()
    n = lentas.size()
    calculadas = {f"{b}_rel": medias[b] / medias["total"] for b in bandas}
    calculadas |= {f"{b}_abs": medias[b] for b in BANDAS_LENTAS}
    calculadas["theta_beta"] = medias["theta"] / medias["beta"]
    for medida, valor in calculadas.items():
        mediana = medianas[medida[:-4]] if medida.endswith("_abs") else np.nan
        salida.append(pd.DataFrame({"valor": valor, "mediana": mediana, "n_epocas": n})
                      .reset_index().assign(medida=medida))
    return pd.concat(salida, ignore_index=True)
