"""Prueba si ICA sirve para sacar las deflexiones de TP9/TP10 (docs/DECISIONES.md).

Por sesión: filtra (LMS + pasa banda 1-40 Hz), ajusta FastICA de 4 componentes sin las
muestras saturadas y anula la componente más correlacionada con la media de TP9 y TP10.
Después compara, en los bloques de tarea, el porcentaje de épocas válidas y la potencia
por banda antes y después. La potencia se compara en las épocas válidas en los dos casos.

Genera outputs/prueba_ica.csv. Tarda unos minutos.

Uso (desde la raíz del repo, con el entorno activado):
    python scripts/prueba_ica.py
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eeg_nback import datos, epocas, espectro, filtrado  # noqa: E402
from eeg_nback.config import load_config  # noqa: E402

UMBRALES = dict(umbral_pp=150.0, umbral_z=3.5, umbral_curtosis=5.0, desvio_min=1.0)
MU = 3e-3


def sin_deflexion(y, fuera_de_saturacion, semilla):
    """Señal sin la componente de ICA más correlacionada con la media de TP9 y TP10."""
    ica = FastICA(n_components=y.shape[1], random_state=semilla, max_iter=1000, tol=1e-3,
                  whiten="unit-variance")
    ica.fit(y[fuera_de_saturacion])
    fuentes = ica.transform(y)
    temporal = (y[:, 0] + y[:, 3]) / 2
    corr = [abs(np.corrcoef(fuentes[fuera_de_saturacion, k], temporal[fuera_de_saturacion])[0, 1])
            for k in range(fuentes.shape[1])]
    fuentes[:, int(np.argmax(corr))] = 0
    return ica.inverse_transform(fuentes), max(corr)


def procesar(sujeto):
    cfg = load_config()
    fs, canales, bandas = cfg["FS"], cfg["CANALES"], cfg["BANDAS"]
    sos = filtrado.disenar_pasabanda(fs)
    t, eeg = datos.cargar_eeg(sujeto["ruta"], canales)
    nback = datos.cargar_nback(sujeto["ruta"])
    bloques = datos.tabla_bloques(nback)

    antes, despues, saturadas, sesiones, correlaciones = [], [], [], [], []
    for ses in datos.tabla_sesiones(sujeto["ruta"], nback).itertuples():
        t_s, x_s = datos.recortar(t, eeg.to_numpy(float), ses.tag_ini, ses.tag_fin)
        sat = epocas.ensanchar_mascara(filtrado.mascara_saturacion(x_s), fs, 1.0)
        y = filtrado.filtrar(x_s, fs, sos, cfg["F_LINEA"], MU)
        limpia, corr = sin_deflexion(y, ~sat.any(axis=1), cfg["SEED"])
        correlaciones.append(corr)
        for b in bloques[bloques["sesion"] == ses.sesion].itertuples():
            antes.append(epocas.recortar_epocas(t_s, y, b.t_ini, b.t_fin, fs)[0])
            despues.append(epocas.recortar_epocas(t_s, limpia, b.t_ini, b.t_fin, fs)[0])
            saturadas.append(epocas.recortar_epocas(t_s, sat, b.t_ini, b.t_fin, fs)[0].any(axis=1))
            sesiones += [ses.sesion] * len(antes[-1])

    antes, despues, saturadas = map(np.concatenate, (antes, despues, saturadas))
    info = pd.DataFrame({"epoca": np.arange(len(antes)), "sesion": sesiones})
    for j, canal in enumerate(canales):
        info[f"sat_{canal}"] = saturadas[:, j]

    def validas(ep):
        tabla = epocas.marcar_artefactos(epocas.tabla_criterios(ep, info, canales), **UMBRALES)
        return tabla.pivot(index="epoca", columns="canal", values="valida")[canales]

    v_antes, v_despues = validas(antes), validas(despues)
    f, p_antes = espectro.psd_welch(np.moveaxis(antes, 1, 2), fs)
    _, p_despues = espectro.psd_welch(np.moveaxis(despues, 1, 2), fs)

    filas = []
    for j, canal in enumerate(canales):
        ambas = (v_antes[canal] & v_despues[canal]).to_numpy()
        fila = {"sujeto": sujeto["sujeto"], "canal": canal, "correlacion": np.median(correlaciones),
                "pct_validas_antes": 100 * v_antes[canal].mean(),
                "pct_validas_ica": 100 * v_despues[canal].mean()}
        for banda, rango in bandas.items():
            fila[banda] = np.nan if ambas.sum() < 20 else np.median(
                espectro.potencia_banda(f, p_despues[ambas, j], rango)
                / espectro.potencia_banda(f, p_antes[ambas, j], rango))
        filas.append(fila)
    print(sujeto["sujeto"], flush=True)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    with Pool(4) as pool:
        tabla = pd.concat(pool.map(procesar, datos.listar_sujetos().to_dict("records"), chunksize=1))
    salida = load_config()["OUTPUT_DIR"] / "prueba_ica.csv"
    tabla.to_csv(salida, index=False)
    print(tabla.groupby("canal")[["pct_validas_antes", "pct_validas_ica", "delta", "theta",
                                  "alfa", "beta"]].median().round(2))
    print(f"-> outputs/{salida.name}")
