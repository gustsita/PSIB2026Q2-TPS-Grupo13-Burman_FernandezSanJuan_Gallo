"""Verifica que estén las dependencias y los datos, y que la sincronización sea consistente.

Uso (desde la raíz del repo, con el entorno activado):
    python scripts/check_env.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PAQUETES = ["numpy", "scipy", "pandas", "statsmodels", "matplotlib", "yaml"]

faltan = []
for p in PAQUETES:
    try:
        importlib.import_module(p)
    except ImportError:
        faltan.append(p)
if faltan:
    sys.exit(f"Faltan paquetes: {faltan}. Correr: pip install -r requirements.txt")
print("Dependencias: OK")

from eeg_nback import datos  # noqa: E402
from eeg_nback.config import load_config  # noqa: E402

cfg = load_config()
try:
    sujetos = datos.listar_sujetos()
except FileNotFoundError as e:
    sys.exit(str(e))

print(f"DATA_DIR: {cfg['DATA_DIR']}")
print(f"Sujetos encontrados: {len(sujetos)} "
      f"(experimento 1: {(sujetos.experimento == 1).sum()}, experimento 2: {(sujetos.experimento == 2).sum()})")

problemas = 0
for _, s in sujetos.iterrows():
    if not s.tiene_eeg:
        print(f"  {s.sujeto}: falta EEG_recording.csv")
        problemas += 1
        continue
    try:
        ses = datos.tabla_sesiones(s.ruta)
    except Exception as e:  # noqa: BLE001
        print(f"  {s.sujeto}: {e}")
        problemas += 1
        continue
    estado = "OK" if ses["dentro"].all() else "estímulos fuera de los tags"
    problemas += estado != "OK"
    print(f"  {s.sujeto}: {len(ses)} sesiones ({', '.join(ses.condicion)}) – {estado}")

print("Todo en orden." if problemas == 0 else f"{problemas} sujeto(s) con problemas.")
