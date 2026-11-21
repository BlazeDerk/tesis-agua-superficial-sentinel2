"""Comprueba las 59 probabilidades ecuatorianas y las 256 externas de la entrega."""
import argparse
import csv
import hashlib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
EXTERNAL_SHA = "18fc33daecad950b3fb25a10988b4d11f4131c8435165e788dc076585b098a8c"


def digest(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def comprobar(value, message):
    if not value:
        raise ValueError(message)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("entrega", type=Path, help="Carpeta Entrega_revision_final_v1")
    cli.add_argument("--zip-externo", type=Path, help="Ubicación alternativa del mismo ZIP")
    args = cli.parse_args()
    external = args.zip_externo or args.entrega / "evaluacion_externa_previa/resultados_externos_v1.zip"
    with (HERE / "inventario_probabilidades.csv").open(encoding="utf-8-sig", newline="") as stream:
        items = list(csv.DictReader(stream))
    with external.open("rb") as stream:
        comprobar(digest(stream) == EXTERNAL_SHA, f"SHA-256 distinto para {external.name}")
    historic = {r["archivo"]: r for r in items if r["evaluacion"] == "historica_20261121"}
    external_rows = {
        str(PurePosixPath(r["directorio"]) / r["archivo"]): r
        for r in items if r["evaluacion"] == "externa_20261120"
    }
    comprobar((len(items), len(historic), len(external_rows)) == (315, 59, 256), "Inventario incompleto o repetido")
    hist_dir = args.entrega / "resultados_hist59_nueva_inferencia_v1/probabilidades"
    comprobar({p.name for p in hist_dir.glob("*.tif")} == set(historic), "La carpeta histórica difiere del inventario")
    for name, record in historic.items():
        path = hist_dir / name
        with path.open("rb") as stream:
            comprobar(path.stat().st_size == int(record["bytes"]) and digest(stream) == record["sha256"], name)
    with ZipFile(external) as archive:
        entries = [name for name in archive.namelist() if name.endswith(".tif")]
        comprobar(len(entries) == 256 and set(entries) == set(external_rows), "El ZIP contiene otros GeoTIFF")
        for name, record in external_rows.items():
            with archive.open(name) as stream:
                comprobar(archive.getinfo(name).file_size == int(record["bytes"]) and digest(stream) == record["sha256"], name)
    print("Se comprobaron el ZIP externo y los 315 GeoTIFF de probabilidades (59 históricos y 256 externos).")


if __name__ == "__main__":
    main()
