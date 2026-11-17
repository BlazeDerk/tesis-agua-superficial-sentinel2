"""FASE 2: recorte y unificacion reproducibles, sin entrenar ni inferir etiquetas.

Uso (despues de descargar TODOS los TIFF principales de GEE):
  python fase2_recortar_unificar.py --legacy-zero-policy excluir
  python fase2_recortar_unificar.py --legacy-zero-policy conservar-con-flags

El script comprueba las tres zonas antes de crear una salida. No sobrescribe
originales ni un destino existente. Los ceros de cuatro canales en El Oro son
ambiguos: su politica debe ser explicita. No se convierten en NoData inferido.
Los bloques geograficos del manifiesto son insumos para una futura division
espacial; NO se asigna train/val/test ni se afirma independencia entre vecinos.
La reexportacion de El Oro pendiente debe auditarse antes de sustituir el
legado; este script no integra automaticamente esa nueva fuente.

Opcional --inventory JSON: {"files":[{"path":"nombre.tif","sha256":"..."}]}.
El inventario debe listar todos los TIFF principales esperados, relativos a
--inputs. Sin un inventario externo no se puede probar que no falta una parte
de un mosaico; solamente la presencia y el contrato de las tres zonas.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / "auditoria_20260930" / "python_deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import rasterio
from rasterio.windows import Window

PATCH = 256
MIN_WATER = 50
NODATA = 65535
CRS = rasterio.crs.CRS.from_epsg(32717)
BANDS = ("B2", "B3", "B4", "B8")
ZONES = ("golfo_guayaquil", "lagos_imbabura", "rio_napo")
EPS = 1e-7
MANIFEST_FIELDS = (
    "patch_id", "kind", "zone", "image", "mask", "source_image", "source_mask",
    "source_image_sha256", "source_mask_sha256", "source_row", "source_col",
    "transform", "bounds", "crs", "water_pixels", "water_fraction",
    "all_zero_pixels", "flags", "geographic_id", "spatial_block_id",
    "touching_group_id", "pair_array_sha256", "output_image_sha256",
    "output_mask_sha256", "label_type", "band_order_evidence",
)


class ContractError(ValueError):
    """Una entrada no cumple el contrato; no se publica una salida parcial."""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_hash(x, y) -> str:
    digest = hashlib.sha256()
    for array in (x, y):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ContractError(message)


def affine_values(transform):
    return list(transform)[:6]


def raster_contract(dataset, count, name, *, mosaic=False):
    require(dataset.count == count, f"{name}: se requieren {count} bandas, hay {dataset.count}.")
    require(set(dataset.dtypes) == {"uint16"}, f"{name}: todas las bandas deben ser uint16.")
    require(dataset.crs == CRS, f"{name}: CRS debe ser EPSG:32717; recibido {dataset.crs}.")
    tr = dataset.transform
    require(np.allclose([tr.a, tr.b, tr.d, tr.e], [10, 0, 0, -10], atol=EPS, rtol=0),
            f"{name}: se requiere malla norte-arriba de 10 m sin rotacion.")
    require(dataset.width > 0 and dataset.height > 0, f"{name}: raster vacio.")
    if mosaic:
        require(dataset.nodata == NODATA, f"{name}: NoData declarado debe ser 65535.")
        require(abs(tr.c / 10 - round(tr.c / 10)) < EPS
                and abs((10000000 - tr.f) / 10 - round((10000000 - tr.f) / 10)) < EPS,
                f"{name}: origen no alineado con la malla de fase1_gee.js.")
    descriptions = tuple(dataset.descriptions)
    expected = BANDS + ("water_mask",) if count == 5 else BANDS if count == 4 else ("water_mask",)
    if any(descriptions):
        require(descriptions == expected, f"{name}: descripciones de bandas inesperadas {descriptions}.")
    return "embedded_band_descriptions" if all(descriptions) else (
        "fase1_gee_contract_not_embedded" if mosaic else "user_context_not_embedded")


def boxes_overlap(a, b):
    return min(a[2], b[2]) - max(a[0], b[0]) > EPS and min(a[3], b[3]) - max(a[1], b[1]) > EPS


def boxes_touch(a, b):
    return max(a[0], b[0]) <= min(a[2], b[2]) + EPS and max(a[1], b[1]) <= min(a[3], b[3]) + EPS


def discover_mosaics(inputs: Path, inventory: Path | None):
    require(inputs.is_dir(), f"No existe la carpeta de entradas: {inputs}")
    sources = []
    ignored_qc = []
    for path in sorted(inputs.rglob("*"), key=lambda p: str(p).casefold()):
        if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
            continue
        if "_QC_" in path.stem.upper():
            ignored_qc.append(str(path.resolve()))
            continue
        matches = [zone for zone in ZONES if path.stem.startswith(f"S2_{zone}_2020_2023")]
        if matches:
            sources.append({"path": path.resolve(), "zone": matches[0]})
    present = {source["zone"] for source in sources}
    missing = sorted(set(ZONES) - present)
    require(not missing, "Faltan mosaicos principales de estas zonas: " + ", ".join(missing)
            + ". No se creo Dataset_UNet_Unificado ni se recortaron datos reales.")
    if inventory:
        spec = json.loads(inventory.read_text(encoding="utf-8-sig"))
        items = spec.get("files", [])
        require(items and isinstance(items, list), "Inventario: files debe ser una lista no vacia.")
        expected = {}
        for item in items:
            relative = Path(item["path"])
            require(not relative.is_absolute(), "Inventario: paths deben ser relativos a inputs.")
            target = (inputs / relative).resolve()
            require(target.is_relative_to(inputs.resolve()), "Inventario: path sale de la carpeta inputs.")
            require(target not in expected, "Inventario: path repetido.")
            require(len(item.get("sha256", "")) == 64, "Inventario: se requiere SHA256 de cada parte.")
            expected[target] = item["sha256"].lower()
        actual = {source["path"] for source in sources}
        require(actual == set(expected), "Inventario y TIFF principales encontrados difieren; faltan o sobran partes.")
        for source in sources:
            require(file_hash(source["path"]) == expected[source["path"]],
                    f"Inventario: SHA256 diferente para {source['path']}.")
    return sources, ignored_qc


def inspect_mosaics(sources, require_qc=False):
    seen_hashes = {}
    for source in sources:
        path = source["path"]
        with rasterio.open(path) as dataset:
            evidence = raster_contract(dataset, 5, path.name, mosaic=True)
            shard = re.search(r"(-\d{10}-\d{10})$", path.stem)
            shard_suffix = shard.group(1) if shard else ""
            prefix = path.stem[:-len(shard_suffix)] if shard_suffix else path.stem
            qc_path = path.with_name(prefix + "_QC_clear_count" + shard_suffix + path.suffix)
            require(not require_qc or qc_path.is_file(),
                    f"{path.name}: falta QC_clear_count de esta parte; se requiere para validar cobertura.")
            qc = rasterio.open(qc_path) if require_qc and qc_path.is_file() else None
            if qc is not None:
                try:
                    require(qc.count == 1 and set(qc.dtypes) == {"uint16"}
                            and qc.nodata == NODATA and qc.crs == dataset.crs,
                            f"{qc_path.name}: QC debe ser una banda uint16 con CRS igual y NoData65535.")
                    require((qc.width, qc.height) == (dataset.width, dataset.height)
                            and qc.transform.almost_equals(dataset.transform, precision=EPS),
                            f"{qc_path.name}: QC no coincide con la malla y dimensiones del mosaico.")
                    require(not any(qc.descriptions) or qc.descriptions == ("clear_count",),
                            f"{qc_path.name}: descripcion QC inesperada.")
                    insufficient = 0
                    for _, window in dataset.block_windows(5):
                        product_valid = np.all(dataset.read_masks(window=window) > 0, axis=0)
                        observations = qc.read(1, window=window)
                        qc_valid = qc.read_masks(1, window=window) > 0
                        insufficient += int(np.count_nonzero(product_valid &
                            (~qc_valid | (observations == NODATA) | (observations < 3))))
                    require(insufficient == 0,
                            f"{path.name}: {insufficient} pixeles marcados validos tienen QC invalido o <3 "
                            "observaciones. Use las copias corregidas por QC; no confundir relleno con fondo.")
                    source.update({"qc_path": str(qc_path), "qc_sha256": file_hash(qc_path),
                                   "qc_verification": "all_jointly_valid_product_pixels_have_at_least_3_clear_observations"})
                finally:
                    qc.close()
            else:
                source["qc_verification"] = "not_requested_not_verified"
            for _, window in dataset.block_windows(5):
                values = dataset.read(5, window=window)
                require(bool(np.all(np.isin(values, [0, 1, NODATA]))),
                        f"{path.name}: water_mask contiene valores distintos de 0, 1 y NoData65535.")
            source.update({"bounds": list(dataset.bounds), "width": dataset.width,
                           "height": dataset.height, "transform": affine_values(dataset.transform),
                           "band_order_evidence": evidence, "sha256": file_hash(path)})
        require(source["sha256"] not in seen_hashes,
                f"Mosaico duplicado byte a byte: {path}; {seen_hashes.get(source['sha256'])}.")
        seen_hashes[source["sha256"]] = str(path)
    for index, source in enumerate(sources):
        for previous in sources[:index]:
            require(not boxes_overlap(source["bounds"], previous["bounds"]),
                    f"Partes de mosaicos superpuestas: {source['path']} y {previous['path']}. "
                    "No se elige silenciosamente una exportacion sobre otra.")


def legacy_pairs(legacy: Path, expected_count: int):
    image_dir, mask_dir = legacy / "images", legacy / "masks"
    require(image_dir.is_dir() and mask_dir.is_dir(), f"Legacy requiere images/ y masks/: {legacy}")
    def listing(folder):
        paths = [p.resolve() for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]
        require(len({p.stem.casefold() for p in paths}) == len(paths), f"{folder}: nombres de parche ambiguos.")
        return {p.stem.casefold(): p for p in paths}
    images, masks = listing(image_dir), listing(mask_dir)
    require(set(images) == set(masks), "Legacy: parejas images/masks incompletas; no emparejar por orden.")
    require(len(images) == expected_count,
            f"Legacy: se esperaban {expected_count} parejas, se encontraron {len(images)}.")
    return [(images[key], masks[key]) for key in sorted(images)]


def candidate(kind, zone, image_path, mask_path, x, y, transform, row, col, image_sha, mask_sha, evidence):
    bounds = list(rasterio.transform.array_bounds(PATCH, PATCH, transform))
    geo = hashlib.sha256((str(CRS) + ":" + ",".join(f"{v:.8f}" for v in affine_values(transform))).encode()).hexdigest()[:20]
    zeros = int(np.count_nonzero(np.all(x == 0, axis=0)))
    flags = []
    if zeros:
        flags.append("all_four_channels_zero_origin_unverified")
    if evidence != "embedded_band_descriptions":
        flags.append("band_order_asserted_not_embedded")
    return {
        "patch_id": f"{kind}_{zone}_{geo}", "kind": kind, "zone": zone,
        "source_image": str(image_path), "source_mask": str(mask_path) if mask_path else str(image_path),
        "source_image_sha256": image_sha, "source_mask_sha256": mask_sha or image_sha,
        "source_row": row, "source_col": col, "transform": affine_values(transform),
        "bounds": bounds, "crs": str(CRS), "water_pixels": int(np.count_nonzero(y == 1)),
        "water_fraction": float(np.mean(y == 1)), "all_zero_pixels": zeros, "flags": flags,
        "geographic_id": geo,
        "spatial_block_id": "epsg32717_10240m_{}_{}".format(math.floor((bounds[0] + bounds[2]) / 20480),
                                                            math.floor((bounds[1] + bounds[3]) / 20480)),
        "pair_array_sha256": pair_hash(x, y), "label_type": "MNDWI_pseudolabel",
        "band_order_evidence": evidence,
    }


def collect_legacy(pairs, policy, counts):
    rows = []
    ambiguous = 0
    for image_path, mask_path in pairs:
        with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask:
            evidence = raster_contract(image, 4, image_path.name)
            raster_contract(mask, 1, mask_path.name)
            require((image.width, image.height) == (PATCH, PATCH)
                    and (mask.width, mask.height) == (PATCH, PATCH),
                    f"{image_path.name}: pareja legacy debe ser 256x256.")
            require(image.crs == mask.crs and image.transform.almost_equals(mask.transform, precision=EPS),
                    f"{image_path.name}: imagen y mascara no estan georreferenciadas en la misma malla.")
            x, y = image.read(), mask.read(1)
            valid_y = mask.read_masks(1) > 0
            require(bool(np.all(np.isin(y[valid_y], [0, 1]))), f"{mask_path.name}: mascara valida no binaria.")
            zero_pixels = int(np.count_nonzero(np.all(x == 0, axis=0)))
            if zero_pixels:
                ambiguous += 1
            if np.any(image.read_masks() == 0) or np.any(~valid_y) or np.any(x == NODATA) or np.any(y == NODATA):
                counts["legacy_nodata"] += 1
                continue
            if int(np.count_nonzero(y == 1)) < MIN_WATER:
                counts["legacy_water_below_50"] += 1
                continue
            if zero_pixels and policy == "excluir":
                counts["legacy_ambiguous_zeros_excluded"] += 1
                continue
            rows.append(candidate("legacy", "el_oro", image_path, mask_path, x, y,
                                  image.transform, 0, 0, file_hash(image_path), file_hash(mask_path), evidence))
    require(not ambiguous or policy is not None,
            f"Hay {ambiguous} parejas legacy con pixeles de cuatro ceros. "
            "Seleccione --legacy-zero-policy excluir o conservar-con-flags; su origen no esta demostrado.")
    counts["legacy_pairs_with_ambiguous_zeros"] = ambiguous
    counts["legacy_candidates"] = len(rows)
    return rows


def collect_new(sources, counts):
    rows = []
    anchors = {zone: (min(source["transform"][2] for source in sources if source["zone"] == zone),
                      max(source["transform"][5] for source in sources if source["zone"] == zone))
               for zone in ZONES}
    for source in sources:
        path = source["path"]
        with rasterio.open(path) as dataset:
            # Ancla comun por zona: fileDimensions8192 es multiplo256 pero el
            # origen del mosaico GEE no tiene por que ser multiplo256 UTMglobal.
            # Esto conserva parches en bordes internos de shards adyacentes.
            # Los IDs siguen basados en transform, independientes del nombre.
            anchor_x, anchor_y = anchors[source["zone"]]
            col_base = int(round((dataset.transform.c - anchor_x) / 10))
            row_base = int(round((anchor_y - dataset.transform.f) / 10))
            col_starts = range((col_base // PATCH) * PATCH, col_base + dataset.width, PATCH)
            row_starts = range((row_base // PATCH) * PATCH, row_base + dataset.height, PATCH)
            for grid_row in row_starts:
                for grid_col in col_starts:
                    row, col = grid_row - row_base, grid_col - col_base
                    counts["new_grid_windows"] += 1
                    if row < 0 or col < 0 or row + PATCH > dataset.height or col + PATCH > dataset.width:
                        counts["new_incomplete_edges"] += 1
                        continue
                    window = Window(col, row, PATCH, PATCH)
                    packed = dataset.read(window=window)
                    x, y = packed[:4], packed[4]
                    if np.any(dataset.read_masks(window=window) == 0) or np.any(packed == NODATA):
                        counts["new_nodata"] += 1
                        continue
                    # Contrato global ya validado: no se construyen etiquetas nuevas.
                    if int(np.count_nonzero(y == 1)) < MIN_WATER:
                        counts["new_water_below_50"] += 1
                        continue
                    rows.append(candidate("new", source["zone"], path, None, x, y,
                                          dataset.window_transform(window), row, col,
                                          source["sha256"], None, source["band_order_evidence"]))
                    counts[f"new_candidates_{source['zone']}"] += 1
    return rows


def deduplicate_and_group(rows, counts):
    kept, geo_seen, content_seen = [], {}, {}
    # Spatial hash: solo compara ventanas cercanas, sin dependencias shapely.
    buckets = {}
    contacts = []
    def keys(bounds):
        return [(x, y) for x in range(math.floor((bounds[0] - EPS) / 10240), math.floor((bounds[2] + EPS) / 10240) + 1)
                for y in range(math.floor((bounds[1] - EPS) / 10240), math.floor((bounds[3] + EPS) / 10240) + 1)]
    for row in rows:
        geo, digest = row["geographic_id"], row["pair_array_sha256"]
        if geo in geo_seen:
            require(geo_seen[geo]["pair_array_sha256"] == digest,
                    f"Etiquetas/imagenes conflictivas para footprint {geo}.")
            counts["duplicate_geographic_pair_skipped"] += 1
            continue
        if digest in content_seen:
            counts["duplicate_content_pair_skipped"] += 1
            continue
        neighbors = set()
        for key in keys(row["bounds"]):
            neighbors.update(buckets.get(key, []))
        for index in sorted(neighbors):
            previous = kept[index]
            require(not boxes_overlap(row["bounds"], previous["bounds"]),
                    f"Parche nuevo/legacy superpuesto: {row['patch_id']} y {previous['patch_id']}. "
                    "Revisar las fuentes para evitar fuga espacial.")
            if boxes_touch(row["bounds"], previous["bounds"]):
                contacts.append((index, len(kept)))
        for key in keys(row["bounds"]):
            buckets.setdefault(key, []).append(len(kept))
        geo_seen[geo], content_seen[digest] = row, row
        kept.append(row)
    parents = list(range(len(kept)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for left, right in contacts:
        parents[find(right)] = find(left)
    groups = {}
    for index, row in enumerate(kept):
        groups.setdefault(find(index), []).append(row["geographic_id"])
    group_ids = {index: "touching_" + hashlib.sha256(
        ":".join(sorted(geos)).encode()).hexdigest()[:16] for index, geos in groups.items()}
    for index, row in enumerate(kept):
        row["touching_group_id"] = group_ids[find(index)]
    return kept, [(kept[a]["patch_id"], kept[b]["patch_id"]) for a, b in contacts]


def load_candidate(row):
    if row["kind"] == "legacy":
        with rasterio.open(row["source_image"]) as image, rasterio.open(row["source_mask"]) as mask:
            x, y, transform = image.read(), mask.read(1), image.transform
            require(image.crs == mask.crs == CRS and image.transform.almost_equals(mask.transform, precision=EPS),
                    f"Fuente cambio durante la ejecucion: {row['patch_id']}")
    else:
        with rasterio.open(row["source_image"]) as image:
            window = Window(row["source_col"], row["source_row"], PATCH, PATCH)
            data = image.read(window=window)
            x, y, transform = data[:4], data[4], image.window_transform(window)
    require(np.allclose(affine_values(transform), row["transform"], rtol=0, atol=EPS)
            and pair_hash(x, y) == row["pair_array_sha256"], f"Fuente cambio durante la ejecucion: {row['patch_id']}")
    return x, y, transform


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (list, dict))
                             else row.get(key, "") for key in fields})


def publish(output, rows, contacts, summary):
    require(not output.exists(), f"Destino ya existe; no se sobrescribe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(f".{output.name}.lock")
    require(not lock.exists(), f"Otra ejecucion o bloqueo pendiente: {lock}")
    staging = output.with_name(f".{output.name}.staging_{uuid.uuid4().hex}")
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        require(not output.exists(), f"Destino ya existe; no se sobrescribe: {output}")
        (staging / "images").mkdir(parents=True)
        (staging / "masks").mkdir()
        for row in rows:
            x, y, transform = load_candidate(row)
            relative_image, relative_mask = f"images/{row['patch_id']}.tif", f"masks/{row['patch_id']}.tif"
            image_path, mask_path = staging / relative_image, staging / relative_mask
            if row["kind"] == "legacy":
                # Copia byte a byte: no se atribuyen metadatos desconocidos a El Oro.
                require(file_hash(Path(row["source_image"])) == row["source_image_sha256"]
                        and file_hash(Path(row["source_mask"])) == row["source_mask_sha256"],
                        f"Fuente legacy cambio durante la ejecucion: {row['patch_id']}")
                shutil.copy2(row["source_image"], image_path)
                shutil.copy2(row["source_mask"], mask_path)
            else:
                profile = {"driver": "GTiff", "height": PATCH, "width": PATCH,
                           "dtype": "uint16", "crs": CRS, "transform": transform,
                           "nodata": NODATA, "compress": "deflate", "tiled": True,
                           "blockxsize": PATCH, "blockysize": PATCH}
                for path, data, descriptions in ((image_path, x, BANDS), (mask_path, y[None], ("water_mask",))):
                    with rasterio.open(path, "w", count=len(descriptions), **profile) as target:
                        target.write(data)
                        target.descriptions = descriptions
                        target.update_tags(source_path=row["source_image"], source_sha256=row["source_image_sha256"],
                                           source_window=f"{row['source_row']},{row['source_col']},256,256",
                                           label_type="MNDWI_pseudolabel", band_order_evidence=row["band_order_evidence"])
            row.update({"image": relative_image, "mask": relative_mask,
                        "output_image_sha256": file_hash(image_path), "output_mask_sha256": file_hash(mask_path)})
        write_csv(staging / "manifest.csv", rows, MANIFEST_FIELDS)
        write_csv(staging / "spatial_contacts.csv", [{"patch_a": a, "patch_b": b} for a, b in contacts], ("patch_a", "patch_b"))
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (staging / "LEEME.txt").write_text(
            "FASE 2 realizada con fuentes locales, sin entrenamiento.\n"
            "NoData65535 se excluye; cero espectral no se infiere como NoData.\n"
            "Las etiquetas MNDWI son pseudoetiquetas, no verdad validada independiente.\n"
            "No dividir aleatoriamente vecinos: usar transform/bounds, touching_group_id y spatial_contacts.csv; "
            "planificar separacion geografica y buffer en la futura fase3.\n"
            "spatial_block_id etiqueta bloques de 10.24 km y no garantiza independencia.\n"
            "Sin inventario SHA256 externo no puede probarse la totalidad de shards descargados.\n", encoding="utf-8")
        require(not output.exists(), f"Destino aparecio durante ejecucion: {output}")
        staging.rename(output)
    finally:
        if staging.exists():
            # Antes del borrado recursivo se comprueba el destino absoluto acotado.
            resolved = staging.resolve()
            require(resolved.parent == output.parent.resolve()
                    and resolved.name.startswith(f".{output.name}.staging_"), "Ruta staging insegura; no se elimina.")
            shutil.rmtree(resolved)
        lock.unlink(missing_ok=True)


def run(args):
    inputs, legacy, output = args.inputs.resolve(), args.legacy.resolve(), args.output.resolve()
    require(not output.exists(), f"Destino ya existe; no se sobrescribe: {output}")
    require(not output.is_relative_to(inputs) and not output.is_relative_to(legacy)
            and not inputs.is_relative_to(output) and not legacy.is_relative_to(output),
            "Destino y carpetas de entrada no pueden contenerse entre si.")
    sources, ignored_qc = discover_mosaics(inputs, args.inventory)
    inspect_mosaics(sources, require_qc=getattr(args, "require_qc", False))
    pairs = legacy_pairs(legacy, args.expected_legacy_count)
    counts = Counter()
    legacy_rows = collect_legacy(pairs, args.legacy_zero_policy, counts)
    new_rows = collect_new(sources, counts)
    rows, contacts = deduplicate_and_group(legacy_rows + new_rows, counts)
    require(bool(rows), "No quedaron parches utiles; no se publica una salida vacia.")
    final_counts = Counter(row["zone"] for row in rows)
    summary = {
        "phase": 2, "training_executed": False, "inputs": str(inputs), "legacy": str(legacy), "output": str(output),
        "patch_size": PATCH, "minimum_water_pixels": MIN_WATER, "no_data": NODATA, "crs": str(CRS),
        "legacy_zero_policy": args.legacy_zero_policy, "legacy_pairs_found": len(pairs),
        "counts": dict(sorted(counts.items())), "accepted_by_zone": dict(sorted(final_counts.items())),
        "accepted_total": len(rows), "accepted_legacy": sum(row["kind"] == "legacy" for row in rows),
        "accepted_new": sum(row["kind"] == "new" for row in rows),
        "all_zero_pairs_retained": sum(row["all_zero_pixels"] > 0 for row in rows),
        "water_pixels": sum(row["water_pixels"] for row in rows),
        "water_fraction_global": sum(row["water_pixels"] for row in rows) / (len(rows) * PATCH * PATCH),
        "touching_pairs": len(contacts), "touching_groups": len({row["touching_group_id"] for row in rows}),
        "spatial_overlaps_accepted": 0, "split_assigned": False,
        "completeness": "sha256_inventory_verified" if args.inventory else "three_zones_present_shard_total_unverified",
        "inventory": str(args.inventory.resolve()) if args.inventory else None, "ignored_qc_files": ignored_qc,
        "qc_required": getattr(args, "require_qc", False),
        "sources": [{key: str(value) if isinstance(value, Path) else value for key, value in source.items()} for source in sources],
        "warnings": ["MNDWI_pseudolabels_require_independent_validation",
                     "legacy_band_order_and_dates_depend_on_user_context",
                     "no_cloud_inference_from_four_optical_channels",
                     "geographic_grouping_and_buffer_required_before_train_val_test_split"],
    }
    publish(output, rows, contacts, summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--inputs", type=Path, default=ROOT / "entradas_fase1")
    result.add_argument("--legacy", type=Path, default=ROOT / "auditoria_20260930" / "corregido")
    result.add_argument("--output", type=Path, default=ROOT / "Dataset_UNet_Unificado")
    result.add_argument("--legacy-zero-policy", choices=("excluir", "conservar-con-flags"))
    result.add_argument("--expected-legacy-count", type=int, default=555,
                        help="Contrato default555; override solo para pruebas/dataset legacy documentado.")
    result.add_argument("--inventory", type=Path, help="Inventario opcional externo de TODAS las partes TIFF y sus SHA256.")
    result.add_argument("--require-qc", action="store_true",
                        help="Exigir QC alineado y >=3 observaciones en todo pixel valido; no recortar relleno exterior como fondo.")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        require(args.expected_legacy_count >= 0, "expected-legacy-count no puede ser negativo.")
        summary = run(args)
    except (ContractError, OSError, ValueError, KeyError, rasterio.errors.RasterioError) as error:
        print(f"FASE 2 NO EJECUTADA/NO PUBLICADA: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": summary["output"], "accepted_total": summary["accepted_total"],
                      "accepted_legacy": summary["accepted_legacy"], "accepted_new": summary["accepted_new"],
                      "completeness": summary["completeness"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
