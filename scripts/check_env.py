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

def extremos_eeg(ruta):
    """Primer y último timestamp del EEG, leyendo solo el principio y el final del archivo.

    Sirve para detectar descargas cortadas sin cargar los ~70 MB. Devuelve None si no se
    pueden leer.
    """
    try:
        with open(ruta, "rb") as f:
            f.readline()                                   # encabezado
            primero = float(f.readline().split(b",")[0])
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            lineas = [l for l in f.read().splitlines() if l.count(b",") == 5]
            ultimo = float(lineas[-1].split(b",")[0])
        return primero, ultimo
    except (ValueError, IndexError):
        return None


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
    t_eeg = extremos_eeg(s.ruta / "EEG_recording.csv")
    if not ses["dentro"].all():
        estado = "estímulos fuera de los tags"
    elif t_eeg is None or t_eeg[0] > ses["tag_ini"].min() or t_eeg[1] < ses["tag_fin"].max():
        estado = "el EEG no cubre todas las sesiones (¿descarga incompleta? volver a correr descargar_datos.py)"
    else:
        estado = "OK"
    problemas += estado != "OK"
    print(f"  {s.sujeto}: {len(ses)} sesiones ({', '.join(ses.condicion)}) – {estado}")

print("Todo en orden." if problemas == 0 else f"{problemas} sujeto(s) con problemas.")
