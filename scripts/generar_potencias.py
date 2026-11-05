"""Genera outputs/potencias_epocas.csv y outputs/potencias_condiciones.csv.

Aplica la cadena completa a los 20 sujetos, en paralelo. Tarda unos minutos. Las tablas
las usan el Notebook 03 (sección 5) y las Partes 4 y 5.

Uso (desde la raíz del repo, con el entorno activado):
    python scripts/generar_potencias.py
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eeg_nback import datos, potencias  # noqa: E402
from eeg_nback.config import load_config  # noqa: E402


def procesar(sujeto):
    cfg = load_config()
    tabla = potencias.tabla_epocas(sujeto["ruta"], cfg["FS"], cfg["CANALES"], cfg["BANDAS"])
    print(sujeto["sujeto"], flush=True)
    return tabla.assign(sujeto=sujeto["sujeto"], experimento=sujeto["experimento"])


if __name__ == "__main__":
    cfg = load_config()
    with Pool(4) as pool:
        tabla = pd.concat(pool.map(procesar, datos.listar_sujetos().to_dict("records"), chunksize=1),
                          ignore_index=True)
    condiciones = potencias.tabla_condiciones(tabla, cfg["BANDAS"])
    tabla.to_csv(cfg["OUTPUT_DIR"] / "potencias_epocas.csv", index=False)
    condiciones.to_csv(cfg["OUTPUT_DIR"] / "potencias_condiciones.csv", index=False)
    print(f"{len(tabla)} filas de épocas y {len(condiciones)} de condiciones -> outputs/")
