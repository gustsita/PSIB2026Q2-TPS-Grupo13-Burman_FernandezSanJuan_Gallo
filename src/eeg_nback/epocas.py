"""Parte 2 – Recorte en realizaciones (épocas) y rechazo de artefactos.

Idea: dentro de un tramo "estable" (un bloque de 1-back, de 3-back o una pausa) el EEG
se puede tratar como un proceso estocástico estacionario en sentido amplio. Cada época
de 2-4 s es una realización de ese proceso. Las épocas contaminadas se descartan antes
de estimar el espectro, porque un solo parpadeo (cientos de µV) domina la PSD.

Épocas de 2 s alineadas con el inicio del bloque: los estímulos se presentan cada 2 s
(500 ms de estímulo + 1500 ms de fijación), así que cada época contiene un ensayo
completo y todas las épocas de la tarea tienen la misma estructura (un estímulo, una
posible respuesta). La resolución de Welch con una ventana de 2 s es 0,5 Hz, suficiente
para bandas de 4 Hz o más de ancho.

Criterios de rechazo (por canal y por época):
1. Amplitud pico a pico mayor que un umbral (parpadeos, movimiento, saturación).
2. Varianza mucho mayor que la del resto de las épocas del mismo canal y sujeto. Se usa
   un z-score robusto del log de la varianza: (v − mediana) / (1,4826 · MAD), para que
   los artefactos no "inflen" el desvío con el que se los compara.
3. Curtosis alta: un parpadeo es un evento aislado de gran amplitud, que hace que la
   distribución de la época tenga colas pesadas (curtosis >> 3).
4. Señal casi plana (desvío muy chico): electrodo sin contacto.
5. Muestras saturadas (±1000 µV), marcadas sobre la señal CRUDA con
   filtrado.mascara_saturacion antes de filtrar. El filtrado desparrama el escalón a las
   muestras vecinas (el pasa banda de 1 Hz "suena" unos 1-2 s y el LMS tarda τ = 1,3 s
   en volver), así que la máscara se ensancha `margen` segundos para cada lado
   (ensanchar_mascara) antes de asignarla a las épocas.

Dos niveles de rechazo. En TP9 y TP10 hay deflexiones lentas de cientos de µV,
simultáneas en los dos canales (en B1, en el 79 % de las épocas). Son compatibles con
parpadeos vistos a través de la referencia Fpz: AF7 y AF8 están al lado de la
referencia y el parpadeo casi se cancela; en los temporales aparece invertido. Esas
deflexiones multiplican la potencia en delta y theta (×25 a ×140 en B1) pero no cambian
la de beta (×0,9 a ×1,0), porque su energía está por debajo de 8 Hz. Entonces los
criterios se calculan dos veces:
- `valida`: sobre la señal de 1-40 Hz; es la que se usa para delta, theta y alfa;
- `valida_beta`: sobre la misma cadena pero con pasa banda de 13-40 Hz; se usa para
  beta. Sigue detectando saturación, EMG y movimiento, pero no rechaza una época por
  un parpadeo que no afecta a beta.
Así beta no pierde las épocas de los canales temporales, que son los que el artículo
usa para esa banda.

Criterio a nivel canal: picos angostos. En algunos canales (Parte 1, sección 5) hay
oscilaciones angostas en 21,3 y 24 Hz, de hasta 1000 veces el fondo. Son casi
senoidales (curtosis < 3), así que los criterios por época no las detectan, y caen
dentro de beta. Para cada sesión y canal se mide qué fracción de la potencia beta
aportan esos picos (fraccion_picos_beta); si supera `umbral_frac`, el canal no se usa
en esa sesión (marcar_canales).

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
from scipy import ndimage, signal, stats

from . import datos, filtrado

MOTIVOS = ("saturada", "pico_a_pico", "varianza", "curtosis", "plana")

# Frecuencias de los picos angostos que aparecen en algunos canales (Parte 1, sección 5)
F_PICOS = (21.33, 24.0)


def recortar_epocas(t, x, t_ini, t_fin, fs, dur=2.0, solape=0.0):
    """Recorta el tramo [t_ini, t_fin) en épocas de `dur` segundos.

    x: array o DataFrame (muestras × canales), o un vector (una sola columna).
    Devuelve (epocas, t_centro):
    - epocas: array (n_epocas, n_muestras, n_canales);
    - t_centro: hora UNIX del centro de cada época (la usa la Parte 4).
    La última época se descarta si queda incompleta.
    """
    t_tramo, x_tramo = datos.recortar(t, np.asarray(x), t_ini, t_fin)
    if x_tramo.ndim == 1:
        x_tramo = x_tramo[:, None]
    n = int(round(dur * fs))
    paso = max(1, int(round(n * (1 - solape))))
    ini = np.arange(0, len(x_tramo) - n + 1, paso)
    if len(ini) == 0:
        return np.empty((0, n, x_tramo.shape[1])), np.empty(0)
    idx = ini[:, None] + np.arange(n)
    return x_tramo[idx], t_tramo[ini] + dur / 2


def ensanchar_mascara(mascara, fs, margen=1.0):
    """Extiende cada True de la máscara `margen` segundos para cada lado (eje 0)."""
    k = int(round(margen * fs))
    m = np.asarray(mascara, dtype=bool)
    if k == 0 or not m.any():
        return m
    estructura = np.ones((2 * k + 1,) + (1,) * (m.ndim - 1), dtype=bool)
    return ndimage.binary_dilation(m, structure=estructura)


def criterios(epocas, canales=None):
    """DataFrame (n_epocas × n_canales filas) con los criterios de cada época y canal.

    Columnas: epoca, canal, pico_a_pico, varianza, curtosis, desvio.
    La curtosis es la de Pearson (fisher=False): vale 3 para una gaussiana.
    """
    n_ep, _, n_can = epocas.shape
    canales = list(canales) if canales is not None else list(range(n_can))
    return pd.DataFrame({
        "epoca": np.repeat(np.arange(n_ep), n_can),
        "canal": np.tile(canales, n_ep),
        "pico_a_pico": np.ptp(epocas, axis=1).ravel(),
        "varianza": np.var(epocas, axis=1).ravel(),
        "curtosis": stats.kurtosis(epocas, axis=1, fisher=False).ravel(),
        "desvio": np.std(epocas, axis=1).ravel(),
    })


def zscore_robusto(v):
    """(v − mediana) / (1,4826 · MAD). Con datos normales equivale a un z-score común."""
    v = np.asarray(v, dtype=float)
    mad = stats.median_abs_deviation(v, nan_policy="omit")
    return (v - np.nanmedian(v)) / (1.4826 * mad)


def marcar_artefactos(tabla, umbral_pp, umbral_z=3.5, umbral_curtosis=5.0, desvio_min=1.0,
                      por=("canal",)):
    """Agrega a la tabla de `criterios` las columnas z_varianza, motivo y valida.

    - El z-score robusto de la varianza se calcula sobre log(varianza), agrupando por
      `por` (por defecto el canal; con varios sujetos, pasar por=("sujeto", "canal")).
      Se usa el logaritmo porque la varianza es positiva y muy asimétrica: en escala
      lineal la mediana queda cerca de cero y casi nada sale por debajo.
      Solo se rechaza por varianza ALTA: la varianza baja la cubre el criterio "plana".
    - La tabla puede traer una columna `saturada` (la época tenía muestras saturadas en
      la señal cruda); si la trae, también es motivo de rechazo.
    - motivo = el primer criterio que falla, en el orden de MOTIVOS ("" si ninguno);
      valida = motivo vacío.
    """
    tabla = tabla.copy()
    claves = [c for c in por if c in tabla]
    logv = np.log(tabla["varianza"].clip(lower=1e-12))
    tabla["z_varianza"] = logv.groupby([tabla[c] for c in claves]).transform(zscore_robusto) \
        if claves else zscore_robusto(logv)

    falla = {
        "saturada": tabla["saturada"].to_numpy(bool) if "saturada" in tabla else np.zeros(len(tabla), bool),
        "pico_a_pico": (tabla["pico_a_pico"] > umbral_pp).to_numpy(),
        "varianza": (tabla["z_varianza"] > umbral_z).to_numpy(),
        "curtosis": (tabla["curtosis"] > umbral_curtosis).to_numpy(),
        "plana": (tabla["desvio"] < desvio_min).to_numpy(),
    }
    motivo = np.full(len(tabla), "", dtype=object)
    for m in reversed(MOTIVOS):          # el primero de la lista queda último: gana
        motivo[falla[m]] = m
    tabla["motivo"] = motivo
    tabla["valida"] = tabla["motivo"] == ""
    return tabla


def epocas_sujeto(ruta, fs, canales, sos, f_linea=60.0, mu=3e-3, dur=2.0,
                  margen_sat=1.0, incluir_pausas=True):
    """Cadena completa para un sujeto: filtra cada sesión y la recorta en épocas.

    Por cada sesión (entre sus tags): marca la saturación sobre la señal cruda y la
    ensancha `margen_sat` s, aplica filtrado.filtrar (LMS + pasa banda) y recorta en
    épocas de `dur` s cada bloque y, si incluir_pausas, cada pausa entre bloques
    (desde el fin de un bloque hasta el primer estímulo del siguiente).

    `sos` define la banda: disenar_pasabanda(fs) para 1-40 Hz, o
    disenar_pasabanda(fs, 13, 40) para la versión de beta.

    Devuelve (epocas, info, canales_ses):
    - epocas: array (n_epocas, n_muestras, n_canales) de la señal filtrada;
    - info: DataFrame con una fila por época: sesion, condicion, bloque, tipo
      ("1-back", "3-back" o "pausa"), t_centro, epoca (índice en `epocas`) y una
      columna sat_<canal> por canal (True si la época tiene muestras saturadas);
    - canales_ses: una fila por sesión y canal con el pico en cada frecuencia de
      F_PICOS (columnas pico_<f>, filtrado.potencia_relativa_en) y la fracción de la
      potencia beta que aportan (frac_picos_beta), medidos sobre la sesión filtrada.
    """
    t, eeg = datos.cargar_eeg(ruta, canales)
    nback = datos.cargar_nback(ruta)
    bloques = datos.tabla_bloques(nback)
    sesiones = datos.tabla_sesiones(ruta, nback)

    if incluir_pausas:
        siguiente = bloques.groupby("sesion")["t_ini"].shift(-1)
        pausas = bloques.assign(t_ini=bloques["t_fin"], t_fin=siguiente, tipo="pausa",
                                n_estimulos=0, aciertos=np.nan, rt_mediana=np.nan)
        tramos = pd.concat([bloques, pausas.dropna(subset=["t_fin"])], ignore_index=True)
    else:
        tramos = bloques

    todas, filas, picos = [], [], []
    for ses in sesiones.itertuples():
        t_s, x_s = datos.recortar(t, eeg.to_numpy(dtype=float), ses.tag_ini, ses.tag_fin)
        sat = ensanchar_mascara(filtrado.mascara_saturacion(x_s), fs, margen_sat)
        y = filtrado.filtrar(x_s, fs, sos, f_linea, mu)
        for j, c in enumerate(canales):
            fila = {"sesion": ses.sesion, "condicion": ses.condicion, "canal": c}
            for f0 in F_PICOS:
                fila[f"pico_{f0:g}"] = filtrado.potencia_relativa_en(y[:, j], fs, f0)
            fila["frac_picos_beta"] = fraccion_picos_beta(y[:, j], fs)
            picos.append(fila)
        for b in tramos[tramos["sesion"] == ses.sesion].sort_values("t_ini").itertuples():
            ep, tc = recortar_epocas(t_s, y, b.t_ini, b.t_fin, fs, dur)
            ep_sat, _ = recortar_epocas(t_s, sat, b.t_ini, b.t_fin, fs, dur)
            if len(ep) == 0:
                continue
            todas.append(ep)
            fila = pd.DataFrame({"sesion": ses.sesion, "condicion": ses.condicion,
                                 "bloque": b.bloque, "tipo": b.tipo, "t_centro": tc})
            for j, c in enumerate(canales):
                fila[f"sat_{c}"] = ep_sat[:, :, j].any(axis=1)
            filas.append(fila)

    info = pd.concat(filas, ignore_index=True)
    info.insert(0, "epoca", np.arange(len(info)))
    return np.concatenate(todas), info, pd.DataFrame(picos)


def fraccion_picos_beta(x, fs, f_picos=F_PICOS, ancho=1.0, beta=(13.0, 30.0)):
    """Fracción de la potencia beta que aportan los picos angostos de `f_picos`.

    PSD de Welch (ventanas de 4 s). En cada pico, el exceso es la potencia en f0 ± ancho
    menos la del fondo en ese mismo ancho, con el fondo estimado como la mediana de la
    PSD de f0 ± 3 a f0 ± 6 Hz (como en filtrado.potencia_relativa_en). Devuelve
    Σ exceso / potencia en beta (0 = sin picos).
    """
    f, p = signal.welch(np.asarray(x) - np.mean(x), fs=fs, nperseg=int(4 * fs))
    df = f[1] - f[0]
    banda = lambda a, b: (f >= a) & (f < b)  # noqa: E731
    exceso = 0.0
    for f0 in f_picos:
        fondo = np.median(p[banda(f0 - 6, f0 - 3) | banda(f0 + 3, f0 + 6)])
        m = banda(f0 - ancho, f0 + ancho)
        exceso += max(0.0, (p[m] - fondo).sum() * df)
    return exceso / (p[banda(*beta)].sum() * df)


def marcar_canales(canales_ses, umbral_frac):
    """Agrega la columna `usar` (bool): False si frac_picos_beta supera umbral_frac."""
    canales_ses = canales_ses.copy()
    canales_ses["usar"] = canales_ses["frac_picos_beta"] <= umbral_frac
    return canales_ses


def tabla_criterios(epocas, info, canales):
    """criterios() de todas las épocas, con la info de cada época y la columna saturada."""
    tabla = criterios(epocas, canales).merge(
        info.drop(columns=[f"sat_{c}" for c in canales]), on="epoca")
    sat = info.set_index("epoca")[[f"sat_{c}" for c in canales]]
    sat.columns = canales
    tabla["saturada"] = sat.stack().reindex(pd.MultiIndex.from_arrays(
        [tabla["epoca"], tabla["canal"]])).to_numpy()
    return tabla


def resumen_calidad(tabla):
    """Porcentaje de épocas válidas por canal (y por condición, si la tabla la tiene)."""
    claves = [c for c in ("condicion", "canal") if c in tabla]
    return (tabla.groupby(claves)["valida"].mean() * 100).rename("pct_validas").reset_index()
