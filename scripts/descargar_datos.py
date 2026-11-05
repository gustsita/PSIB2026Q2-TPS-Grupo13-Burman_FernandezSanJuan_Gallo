"""Descarga de PhysioNet solo los archivos que usa el TPS (EEG, tags y respuestas del n-back).

El zip completo del dataset pesa unos 585 MB e incluye las señales de las pulseras, que
no usamos. Este script baja, por sujeto, EEG_recording.csv (~70 MB), tags.csv y
n_back_responses.csv, y los guarda en DATA_DIR con la misma estructura que PhysioNet.

Uso (desde la raíz del repo):
    python scripts/descargar_datos.py              # los 20 sujetos (~1,4 GB)
    python scripts/descargar_datos.py B1 A9        # solo algunos sujetos

Si un archivo ya está completo, no se vuelve a bajar; si quedó a medias, se continúa.
Usa curl (viene con macOS y con Windows 10/11) en lugar de urllib, porque el Python de
python.org en macOS suele fallar al verificar los certificados SSL.
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eeg_nback.config import load_config  # noqa: E402

BASE = "https://physionet.org/files/brain-wearable-monitoring/1.0.0"
ARCHIVOS = ("tags.csv", "n_back_responses.csv", "EEG_recording.csv")
SUJETOS = [f"A{i}" for i in range(1, 11)] + [f"B{i}" for i in range(1, 11)]


def carpeta_experimento(sujeto):
    return "Experiment_1" if sujeto.startswith("A") else "Experiment_2"


def tamano_remoto(url):
    """Tamaño en bytes según el encabezado Content-Length (sin bajar el archivo)."""
    cabecera = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True, check=True).stdout
    largos = [linea.split(":")[1] for linea in cabecera.lower().splitlines()
              if linea.startswith("content-length")]
    return int(largos[-1]) if largos else -1


def bajar(url, destino, intentos=20):
    """Baja url a destino. PhysioNet corta conexiones largas o las deja colgadas: si se
    corta, o si pasa 1 minuto a menos de 1 kB/s, reintenta continuando desde donde quedó
    (-C -) hasta que el archivo tenga el tamaño completo."""
    total = tamano_remoto(url)
    for intento in range(1, intentos + 1):
        subprocess.run(["curl", "-L", "--fail", "-C", "-", "-#", "--connect-timeout", "30",
                        "--speed-limit", "1000", "--speed-time", "60", "-o", str(destino), url])
        actual = destino.stat().st_size if destino.exists() else 0
        if actual == total:
            return
        print(f"    se cortó en {actual / 1e6:.1f} / {total / 1e6:.1f} MB; reintento {intento}/{intentos}")
        time.sleep(5)
    sys.exit(f"No se pudo completar {destino.name}. Volver a correr el script: continúa desde donde quedó.")


def main(sujetos):
    data_dir = load_config()["DATA_DIR"]
    for sujeto in sujetos:
        carpeta = data_dir / carpeta_experimento(sujeto) / sujeto
        carpeta.mkdir(parents=True, exist_ok=True)
        for nombre in ARCHIVOS:
            url = f"{BASE}/{carpeta_experimento(sujeto)}/{sujeto}/{nombre}"
            destino = carpeta / nombre
            if destino.exists() and destino.stat().st_size == tamano_remoto(url):
                print(f"{sujeto}/{nombre}: ya está")
                continue
            print(f"{sujeto}/{nombre}: bajando")
            bajar(url, destino)


if __name__ == "__main__":
    elegidos = sys.argv[1:] or SUJETOS
    desconocidos = set(elegidos) - set(SUJETOS)
    if desconocidos:
        sys.exit(f"Sujetos desconocidos: {sorted(desconocidos)}. Válidos: A1-A10, B1-B10.")
    main(elegidos)
