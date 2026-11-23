"""Reconstruye los ocho fragmentos originales del navegador a partir del checkpoint."""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "revision"))
from modelo_original import load_original_model

MANIFEST_SHA256 = "028b7f290401fd2cd2c2c7b3af6488f04f2abfac33597412f1ab875088a2f7aa"


def leaf_layers(layer):
    if hasattr(layer, "layers"):
        for child in layer.layers:
            yield from leaf_layers(child)
    else:
        yield layer


def prepare(checkpoint):
    destination = ROOT / "dist/model"
    raw = (destination / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("El manifiesto difiere de la versión utilizada en el navegador.")
    manifest = json.loads(raw)
    model = load_original_model(checkpoint)
    layers = {layer.name: layer for layer in leaf_layers(model)}
    chunks = {entry["file"]: bytearray(entry["byte_length"]) for entry in manifest["chunks"]}
    for entry in manifest["arrays"]:
        tensor = getattr(layers[entry["layer"]], entry["name"])
        data = np.asarray(tensor.numpy(), dtype="<f4", order="C").tobytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f'Pesos inesperados en {entry["layer"]}/{entry["name"]}.')
        for segment in entry["segments"]:
            offset = segment["byte_offset"]
            source = segment["array_byte_offset"]
            size = segment["byte_length"]
            chunks[segment["chunk"]][offset:offset + size] = data[source:source + size]

    with tempfile.TemporaryDirectory(dir=destination, prefix=".pesos-") as tmp:
        scratch = Path(tmp)
        for entry in manifest["chunks"]:
            name = entry["file"]
            data = chunks[name]
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"Error al reconstruir {name}.")
            (scratch / name).write_bytes(data)
        for name in chunks:
            (scratch / name).replace(destination / name)
    print(f'{len(chunks)} fragmentos verificados, {manifest["total_weight_bytes"]} bytes.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Ruta al model_best.keras original")
    prepare(parser.parse_args().checkpoint)
