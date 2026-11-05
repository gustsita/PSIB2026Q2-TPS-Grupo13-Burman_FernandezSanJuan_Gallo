"""Genera outputs/deflexiones_beta.csv, la tabla que usa el Notebook 03 (sección 4).

Aplica la cadena de filtrado dos veces a los 20 sujetos, en paralelo. Tarda unos minutos.

Uso (desde la raíz del repo, con el entorno activado):
    python scripts/generar_deflexiones.py
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eeg_nback import datos, epocas  # noqa: E402
from eeg_nback.config import load_config  # noqa: E402

UMBRALES = dict(umbral_pp=150.0, umbral_z=3.5, umbral_curtosis=5.0, desvio_min=1.0)


def procesar(sujeto):
    cfg = load_config()
    tabla = epocas.tabla_deflexiones(sujeto["ruta"], cfg["FS"], cfg["CANALES"],
                                     cfg["BANDAS"]["beta"], UMBRALES)
    print(sujeto["sujeto"], flush=True)
    return tabla.assign(sujeto=sujeto["sujeto"], experimento=sujeto["experimento"])


if __name__ == "__main__":
    sujetos = datos.listar_sujetos().to_dict("records")
    with Pool(4) as pool:
        tabla = pd.concat(pool.map(procesar, sujetos, chunksize=1), ignore_index=True)
    salida = load_config()["OUTPUT_DIR"] / "deflexiones_beta.csv"
    tabla.to_csv(salida, index=False)
    print(f"{len(tabla)} filas -> outputs/{salida.name}")
