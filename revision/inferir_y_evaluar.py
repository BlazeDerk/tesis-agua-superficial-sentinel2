"""Evaluate the frozen U-Net and two fixed indices on an external public reference.

Reference provenance is S1S2-Water's NDWI-initialized, expert-refined masks;
optics are matched L2A acquisitions. Importing this module never imports
TensorFlow. --validate-only validates real input files without model inference
or metric outputs. A new output directory is required for every invocation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

VERSION = "1.0.0"
CHECKPOINT_SHA256 = "9e28f92b557bdd2e9a2e7b068a6622254ddbf80080926b8c004368ad5683dda4"
BANDS = ("B2", "B3", "B4", "B8", "B11")
METHODS = ("unet", "ndwi", "mndwi")
COUNT_KEYS = ("tp", "fp", "fn", "tn")
REFERENCE_ORIGIN = "public_expert_refined_ndwi"
REQUIRED = (
    "case_id", "scene_id", "cluster_id", "split", "optical_path",
    "reference_path", "validity_path", "optical_sha256", "reference_sha256",
    "validity_sha256", "row_off", "col_off", "height", "width",
    "reference_origin", "l1c_source", "l2a_product",
)


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows):
    require(bool(rows), "No se escribe una tabla vacía como resultado.")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def binary_metrics(counts):
    tp, fp, fn, tn = (int(counts[key]) for key in COUNT_KEYS)
    definitions = {
        "iou": (tp, tp + fp + fn), "f1": (2 * tp, 2 * tp + fp + fn),
        "precision": (tp, tp + fp), "recall": (tp, tp + fn),
    }
    result = {}
    for name, (numerator, denominator) in definitions.items():
        result[name] = numerator / denominator if denominator else 0.0
        result[name + "_defined"] = bool(denominator)
    result["reference_has_no_water"] = tp + fn == 0
    result["prediction_has_no_water"] = tp + fp == 0
    return result


def confusion(reference, prediction, valid):
    truth = reference == 1
    return {
        "tp": int(np.count_nonzero(valid & truth & prediction)),
        "fp": int(np.count_nonzero(valid & ~truth & prediction)),
        "fn": int(np.count_nonzero(valid & truth & ~prediction)),
        "tn": int(np.count_nonzero(valid & ~truth & ~prediction)),
    }


def evaluate_arrays(optical, reference, validity, probability, *,
                    optical_gdal=None, reference_gdal=None,
                    validity_gdal=None, probability_gdal=None):
    """Evaluate arrays over one explicit common support, without model loading.

    Optical is CHW physical reflectance, not uint16 DN. Array dimensions are
    deliberately generic for meaningful small analytical tests. Raster input
    validation enforces the model's actual 256 x 256 tile size.
    """
    require(optical.ndim == 3 and optical.shape[0] == 5, "Óptica: cinco bandas CHW.")
    shape = optical.shape[1:]
    require(reference.shape == validity.shape == probability.shape == shape,
            "Las dimensiones de los arrays no coinciden.")
    require(np.issubdtype(optical.dtype, np.floating), "La óptica debe ser reflectancia flotante.")
    require(bool(np.all(np.isin(reference, [0, 1]))), "Referencia: solo clases 0 y 1.")
    require(bool(np.all(np.isin(validity, [0, 1]))), "VALID: solo valores 0 y 1.")

    def mask_or_all(value):
        if value is None:
            return np.ones(shape, dtype=bool)
        require(value.shape == shape, "Dimensiones incorrectas de máscara GDAL.")
        return np.asarray(value, dtype=bool)

    optical_gdal = mask_or_all(optical_gdal)
    reference_gdal = mask_or_all(reference_gdal)
    validity_gdal = mask_or_all(validity_gdal)
    probability_gdal = mask_or_all(probability_gdal)
    values = optical.astype(np.float64, copy=False)
    finite = np.isfinite(values).all(axis=0)
    green, nir, swir = values[1], values[3], values[4]
    with np.errstate(invalid="ignore", over="ignore"):
        nd_denom, mnd_denom = green + nir, green + swir
    prob_finite = np.isfinite(probability)
    reasons = {
        "optical_gdal_invalid": ~optical_gdal,
        "optical_nonfinite": ~finite,
        "reference_gdal_invalid": ~reference_gdal,
        "validity_gdal_invalid": ~validity_gdal,
        "manual_validity_not_one": validity != 1,
        "ndwi_denominator_nonpositive": finite & (nd_denom <= 0),
        "mndwi_denominator_nonpositive": finite & (mnd_denom <= 0),
        "probability_gdal_invalid": ~probability_gdal,
        "probability_nonfinite": ~prob_finite,
        "probability_out_of_range": prob_finite & ((probability < 0) | (probability > 1)),
    }
    common = np.ones(shape, dtype=bool)
    control = {"total_pixels": int(common.size)}
    for name, reason in reasons.items():
        control["independent_" + name] = int(reason.sum())
        control["sequential_" + name] = int((common & reason).sum())
        common &= ~reason
    control["common_valid_pixels"] = int(common.sum())
    control["excluded_pixels"] = int(common.size - common.sum())
    control["negative_reflectance_pixels_not_automatically_excluded"] = int(
        (finite & (values < 0).any(axis=0)).sum())
    control["reflectance_gt_one_pixels_not_automatically_excluded"] = int(
        (finite & (values > 1).any(axis=0)).sum())
    require(sum(value for key, value in control.items() if key.startswith("sequential_"))
            == control["excluded_pixels"], "Los motivos secuenciales no concilian.")
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = np.divide(green - nir, nd_denom, out=np.zeros(shape), where=common)
        mndwi = np.divide(green - swir, mnd_denom, out=np.zeros(shape), where=common)
    predictions = {"unet": probability >= 0.5, "ndwi": ndwi > 0, "mndwi": mndwi > 0}
    rows = []
    for method, prediction in predictions.items():
        counts = confusion(reference, prediction, common)
        require(sum(counts.values()) == control["common_valid_pixels"], "Soporte desigual entre métodos.")
        rows.append({"method": method, "valid_pixels": int(common.sum()),
                     "water_reference_pixels": counts["tp"] + counts["fn"],
                     "water_predicted_pixels": counts["tp"] + counts["fp"],
                     **counts, **binary_metrics(counts),
                     "status": "evaluated" if common.any() else "no_common_valid_pixels"})
    return rows, control, common


def same_grid(a, b, label):
    require(a.crs is not None and a.crs == b.crs, f"CRS ausente o distinto: {label}.")
    require(a.shape == b.shape and a.transform == b.transform,
            f"Cuadrícula distinta: {label}; no se remuestrea durante la evaluación.")


def read_case(row, verify_hashes=True):
    sources = {}
    for kind, path in row["_paths"].items():
        actual = sha256(path)
        expected = row[kind + "_sha256"].lower()
        if verify_hashes:
            require(actual == expected, f"SHA-256 no coincide: {path}")
        sources[kind] = {"path": str(path), "sha256": actual}
    with rasterio.open(row["_paths"]["optical"]) as optical, \
            rasterio.open(row["_paths"]["reference"]) as reference, \
            rasterio.open(row["_paths"]["validity"]) as validity:
        require(optical.count == 5 and optical.dtypes == ("float32",) * 5,
                "Óptica: se requieren cinco bandas float32.")
        require(optical.descriptions == BANDS, "Orden/descripción de bandas incorrectos: se exige B2,B3,B4,B8,B11.")
        require(all(value == 1 for value in optical.scales) and all(value == 0 for value in optical.offsets),
                "La óptica debe almacenar reflectancia física; escala/offset deben haberse aplicado una sola vez.")
        require(optical.shape == (256, 256), "Cada caso debe medir 256 × 256 píxeles.")
        require(optical.transform.b == optical.transform.d == 0, "Se requiere cuadrícula sin rotación.")
        require(reference.count == validity.count == 1 and reference.dtypes[0] == validity.dtypes[0] == "uint8",
                "MSK y VALID deben ser rasters de una banda uint8.")
        require(reference.nodata not in (0, 1), "La referencia no puede declarar fondo o agua como NoData.")
        same_grid(optical, reference, "MSK")
        same_grid(optical, validity, "VALID")
        data = optical.read()
        labels, valid = reference.read(1), validity.read(1)
        require(np.all(np.isin(labels, [0, 1])), "MSK contiene valores distintos de 0/1.")
        require(np.all(np.isin(valid, [0, 1])), "VALID contiene valores distintos de 0/1.")
        masks = {"optical_gdal": (optical.read_masks() != 0).all(axis=0),
                 "reference_gdal": reference.read_masks(1) != 0,
                 "validity_gdal": validity.read_masks(1) != 0}
        input_valid = (optical.read_masks()[:4] != 0).all(axis=0) & np.isfinite(data[:4]).all(axis=0)
        profile = optical.profile.copy()
        profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")
        origin_transform = Affine(optical.transform.a, 0,
                                  optical.transform.c - row["col_off"] * optical.transform.a,
                                  0, optical.transform.e,
                                  optical.transform.f - row["row_off"] * optical.transform.e)
        scene_grid = (str(optical.crs), tuple(origin_transform)[:6])
    for kind, path in row["_paths"].items():
        require(sha256(path) == sources[kind]["sha256"], f"Un archivo cambió durante su lectura: {path}")
    return {"optical": data, "reference": labels, "validity": valid, "masks": masks,
            "input_valid": input_valid, "profile": profile, "scene_grid": scene_grid, "sources": sources}


def validate_frozen_plan(manifest, rows, plan_path=None):
    """Bind the full manifest to the frozen selection and acquisition receipts."""
    plan_path = Path(plan_path).resolve() if plan_path else manifest.parent / "plan_congelado.json"
    state_path = manifest.parent / "estado_adquisicion.json"
    require(plan_path.is_file(), "Falta plan_congelado.json; no se ejecuta una selección sin vínculo al plan.")
    require(state_path.is_file(), "Falta estado_adquisicion.json.")
    plan_hash, state_hash = sha256(plan_path), sha256(state_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    require(state.get("complete") is True, "La adquisición está incompleta; no se evalúa el manifiesto parcial.")
    require(state.get("plan_sha256") == plan_hash, "El plan no coincide con el SHA-256 del estado de adquisición.")
    require(state.get("manifest") == manifest.name, "El estado no identifica este manifiesto final.")
    require(state.get("manifest_sha256") == sha256(manifest), "El manifiesto no coincide con el SHA-256 del estado.")
    require(plan.get("split") == "test" and plan.get("reference_origin") == REFERENCE_ORIGIN,
            "El plan no corresponde al Test con referencia experta inicializada por NDWI.")
    planned = plan.get("cases", [])
    scenes = {str(item["scene_id"]): item for item in plan.get("scenes", [])}
    require(bool(planned) and bool(scenes), "El plan no contiene escenas/casos.")
    expected = {item["case_id"]: item for item in planned}
    require(len(expected) == len(planned), "El plan repite identificadores de caso.")
    require(state.get("expected_cases") == len(planned) == len(rows),
            "La cohorte no contiene exactamente todos los casos del plan congelado.")
    require(state.get("expected_scenes") == len(scenes) == len({row["scene_id"] for row in rows}),
            "La cohorte no contiene exactamente todas las escenas previstas.")
    require({row["case_id"] for row in rows} == set(expected), "Los case_id no coinciden con el plan congelado.")
    require(len(scenes) == len(plan["scenes"]), "El plan repite escenas.")
    case_metadata = []
    scene_receipts = []
    for scene_id, scene in scenes.items():
        for path_key, hash_key in (("item_path", "item_sha256"), ("mask_path", "mask_sha256"),
                                   ("valid_path", "valid_sha256")):
            source = Path(scene.get(path_key, ""))
            if not source.is_absolute():
                source = plan_path.parent / source
            require(source.is_file(), f"Falta el insumo enlazado por el plan: {path_key}, escena {scene_id}.")
            require(sha256(source) == scene.get(hash_key), f"Hash de {path_key} no coincide con el plan, escena {scene_id}.")
            scene_receipts.append({"scene_id": scene_id, "kind": path_key, "path": str(source.resolve()),
                                   "sha256": scene[hash_key]})
    for row in rows:
        selected = expected[row["case_id"]]
        for key in ("scene_id", "cluster_id", "row_off", "col_off", "height", "width"):
            require(str(row[key]) == str(selected.get(key)), f"{row['case_id']}: {key} difiere del plan.")
        require(row["scene_id"] in scenes, "Una escena del manifiesto no pertenece al plan.")
        scene = scenes[row["scene_id"]]
        require(row["l1c_source"] == scene.get("l1c_source") and row["l2a_product"] == scene.get("l2a_product"),
                f"{row['case_id']}: los productos L1C/L2A no coinciden con el plan.")
        require(row["cluster_id"] == scene.get("cluster_id"), "Cluster distinto del previsto para la escena.")
        optical_path = Path(row["optical_path"])
        if not optical_path.is_absolute():
            optical_path = manifest.parent / optical_path
        receipt_path = optical_path.parent / (row["case_id"] + ".json")
        require(receipt_path.is_file(), "Faltan los metadatos de adquisición del caso: " + str(receipt_path))
        receipt_hash = sha256(receipt_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
        require(receipt.get("plan_sha256") == plan_hash, "Un caso enlaza otra versión del plan.")
        require(receipt.get("l2a_item_sha256") == scene.get("item_sha256"), "Un caso enlaza otro ítem L2A.")
        require([item.get("band") for item in receipt.get("source_bands", [])] == list(BANDS),
                "Los metadatos no documentan las cinco bandas requeridas.")
        saved_manifest = receipt.get("manifest", {})
        for key in REQUIRED:
            require(str(row[key]) == str(saved_manifest.get(key)),
                    f"{row['case_id']}: {key} difiere de los metadatos de adquisición.")
        case_metadata.append({"case_id": row["case_id"], "path": str(receipt_path.resolve()), "sha256": receipt_hash})
    require(sha256(plan_path) == plan_hash and sha256(state_path) == state_hash,
            "El plan o el estado cambiaron durante la validación.")
    return {"plan_path": str(plan_path), "plan_sha256": plan_hash,
            "acquisition_state_path": str(state_path), "acquisition_state_sha256": state_hash,
            "acquisition_complete": True, "manifest_matches_complete_frozen_cohort": True,
            "scene_source_receipts": scene_receipts, "case_acquisition_metadata": case_metadata}


def validate_manifest(path, plan_path=None):
    manifest = Path(path).resolve()
    require(manifest.is_file(), f"No existe el manifiesto: {manifest}")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(set(REQUIRED).issubset(reader.fieldnames or []), "Faltan columnas obligatorias del manifiesto.")
        rows = [{key: str(value or "").strip() for key, value in row.items()} for row in reader]
    require(bool(rows), "El manifiesto no contiene casos reales.")
    frozen = validate_frozen_plan(manifest, rows, plan_path)
    case_ids, optical_paths = set(), set()
    scenes = {}
    rectangles = defaultdict(list)
    preflight = []
    for index, row in enumerate(rows, 1):
        for key in REQUIRED:
            require(bool(row.get(key)), f"Fila {index}: falta {key}.")
        require(row["split"] == "test", f"Fila {index}: solo se permite split=test.")
        require(row["reference_origin"] == REFERENCE_ORIGIN,
                f"Fila {index}: reference_origin debe ser {REFERENCE_ORIGIN}.")
        require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["case_id"])), "case_id inseguro o vacío.")
        require(row["case_id"] not in case_ids, "case_id duplicado: " + row["case_id"])
        case_ids.add(row["case_id"])
        try:
            for key in ("row_off", "col_off", "height", "width"):
                row[key] = int(row[key])
        except ValueError as error:
            raise ValidationError(f"Fila {index}: posición/dimensiones deben ser enteros.") from error
        require(row["height"] == row["width"] == 256, "El manifiesto debe declarar ventanas de 256 × 256.")
        require(row["row_off"] >= 0 and row["col_off"] >= 0 and
                row["row_off"] % 256 == row["col_off"] % 256 == 0,
                "Los offsets deben pertenecer a la cuadrícula completa de 256.")
        row["_paths"] = {}
        for kind in ("optical", "reference", "validity"):
            item = Path(row[kind + "_path"]).expanduser()
            if not item.is_absolute():
                item = manifest.parent / item
            item = item.resolve()
            require(item.is_file(), f"No existe {kind}: {item}")
            digest = row[kind + "_sha256"].lower()
            require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), f"Hash inválido de {kind}.")
            row["_paths"][kind] = item
        require(row["_paths"]["optical"] not in optical_paths, "Una óptica se cuenta dos veces.")
        optical_paths.add(row["_paths"]["optical"])
        data = read_case(row)
        identity = (row["cluster_id"], row["l1c_source"], row["l2a_product"], data["scene_grid"])
        require(row["scene_id"] not in scenes or scenes[row["scene_id"]] == identity,
                "Una escena declara distintos clusters, productos o cuadrículas originales.")
        scenes[row["scene_id"]] = identity
        rect = (row["row_off"], row["col_off"], row["row_off"] + 256, row["col_off"] + 256)
        for prior in rectangles[row["scene_id"]]:
            require(not (min(rect[2], prior[2]) > max(rect[0], prior[0]) and
                         min(rect[3], prior[3]) > max(rect[1], prior[1])),
                    "Ventanas solapadas o duplicadas dentro de una escena.")
        rectangles[row["scene_id"]].append(rect)
        optical = data["optical"].astype(np.float64, copy=False)
        potential = (data["masks"]["optical_gdal"] & data["masks"]["reference_gdal"] &
                     data["masks"]["validity_gdal"] & (data["validity"] == 1) &
                     np.isfinite(optical).all(axis=0) &
                     ((optical[1] + optical[3]) > 0) & ((optical[1] + optical[4]) > 0))
        # Validation reports support only. It constructs no probability array,
        # calls no model and computes no classification metrics.
        preflight.append({"case_id": row["case_id"], "scene_id": row["scene_id"],
                          "cluster_id": row["cluster_id"], "sources": data["sources"],
                          "row_off": row["row_off"], "col_off": row["col_off"],
                          "height": row["height"], "width": row["width"],
                          "l1c_source": row["l1c_source"], "l2a_product": row["l2a_product"],
                          "potential_common_pixels_before_model": int(potential.sum()),
                          "input_pixels_requiring_zero_filler": int((~data["input_valid"]).sum())})
    require(sum(item["potential_common_pixels_before_model"] for item in preflight) > 0,
            "La cohorte completa carece de soporte evaluable; no se generan métricas con datos vacíos.")
    return manifest, rows, preflight, frozen


def aggregate(rows, scope, key=None):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(str(row[key]) if key else "ALL", row["method"])].append(row)
    result = []
    for (identifier, method), items in sorted(buckets.items()):
        counts = {name: sum(int(item[name]) for item in items) for name in COUNT_KEYS}
        valid = sum(counts.values())
        result.append({"aggregation": scope, "aggregate_id": identifier, "method": method,
                       "n_cases": len(items), "n_scenes": len({item["scene_id"] for item in items}),
                       "n_clusters": len({item["cluster_id"] for item in items}),
                       "n_cases_with_valid_pixels": sum(item["valid_pixels"] > 0 for item in items),
                       "valid_pixels": valid, "water_reference_pixels": counts["tp"] + counts["fn"],
                       "water_predicted_pixels": counts["tp"] + counts["fp"],
                       **counts, **binary_metrics(counts),
                       "status": "evaluated" if valid else "no_common_valid_pixels"})
    return result


def paired_differences(tables):
    differences = []
    for rows in tables:
        by = {(row["aggregate_id"], row["method"]): row for row in rows}
        for identifier in sorted({row["aggregate_id"] for row in rows}):
            unet = by[identifier, "unet"]
            for baseline in ("ndwi", "mndwi"):
                other = by[identifier, baseline]
                require(unet["valid_pixels"] == other["valid_pixels"], "Comparación sin soporte pareado.")
                row = {"aggregation": unet["aggregation"], "aggregate_id": identifier,
                       "comparison": "unet_minus_" + baseline, "valid_pixels": unet["valid_pixels"],
                       "status": unet["status"]}
                for metric in ("iou", "f1", "precision", "recall"):
                    defined = unet[metric + "_defined"] and other[metric + "_defined"]
                    row[metric + "_difference_defined"] = defined
                    row[metric + "_difference_pp"] = 100 * (unet[metric] - other[metric]) if defined else ""
                differences.append(row)
    return differences


def run(manifest_path, checkpoint, out_dir, *, validate_only=False, batch_size=4, plan_path=None):
    require(batch_size > 0, "batch_size debe ser positivo.")
    out = Path(out_dir).resolve()
    require(not out.exists(), "El directorio de salida ya existe; elija uno nuevo para no mezclar ejecuciones.")
    manifest_hash = sha256(Path(manifest_path))
    manifest, rows, preflight, frozen = validate_manifest(manifest_path, plan_path)
    require(sha256(manifest) == manifest_hash, "El manifiesto cambió durante su validación.")
    started = datetime.now(timezone.utc).isoformat()
    common_metadata = {
        "schema_version": VERSION, "started_at_utc": started,
        "manifest_path": str(manifest), "manifest_sha256": manifest_hash,
        "n_cases": len(rows), "n_scenes": len({row["scene_id"] for row in rows}),
        "n_clusters": len({row["cluster_id"] for row in rows}),
        "reference_origin": REFERENCE_ORIGIN,
        "reference_procedure": "NDWI initialization followed by iterative expert review/correction; possible anchoring to NDWI",
        "dataset_adaptation": "S1S2-Water v1.0.1 references and validity with matched same-acquisition Sentinel-2 L2A",
        "optical_contract": "float32 physical surface reflectance; scale/offset applied once upstream; B2/B3/B4/B8/B11",
        "required_checkpoint_sha256": CHECKPOINT_SHA256,
        "thresholds": {"unet": ">=0.5", "ndwi": ">0", "mndwi": ">0"},
        "input_filler": "all four input channels set to zero where any input band is GDAL-invalid or nonfinite; filled pixels excluded",
        "filler_limitation": "Convolutions can propagate filler influence to adjacent valid predictions; no additional dilation was applied",
        "bootstrap_executed": False, "training_executed": False, "threshold_tuning": False,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "rasterio": rasterio.__version__},
        "inputs": preflight,
        "frozen_cohort_verification": frozen,
        "selection_limit": "The manifest supplies the preselected cohort; validation does not certify historical freezing, temporal identity or independence",
    }
    if validate_only:
        if checkpoint:
            from modelo_original import verify_checkpoint
            common_metadata["checkpoint"] = verify_checkpoint(checkpoint)
        out.mkdir(parents=True)
        common_metadata.update(status="validated_inputs_only", inference_executed=False,
                               evaluation_executed=False, metric_outputs_created=False,
                               completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "validacion_paquete.json", common_metadata)
        return common_metadata
    require(checkpoint is not None, "Falta --checkpoint; solo --validate-only permite omitir el modelo.")
    from modelo_original import load_original_model, predict_reflectance_tiles, verify_checkpoint
    checkpoint_record = verify_checkpoint(checkpoint)
    model = load_original_model(checkpoint)
    out.mkdir(parents=True)
    probability_dir = out / "probabilidades"
    probability_dir.mkdir()
    per_case, controls, probability_sources = [], [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        data_chunk = [read_case(row) for row in chunk]
        inputs = []
        for data in data_chunk:
            array = np.moveaxis(data["optical"][:4], 0, -1).copy()
            array[~data["input_valid"]] = 0
            inputs.append(array)
        batch = np.ascontiguousarray(np.stack(inputs), dtype=np.float32)
        probabilities = predict_reflectance_tiles(model, batch)
        require(probabilities.shape == (len(chunk), 256, 256, 1), "Forma de probabilidades inesperada.")
        require(np.issubdtype(probabilities.dtype, np.floating), "La probabilidad debe ser flotante.")
        for row, data, prediction in zip(chunk, data_chunk, probabilities):
            probability = np.asarray(prediction[:, :, 0], dtype=np.float32)
            probability_valid = data["input_valid"] & np.isfinite(probability) & (probability >= 0) & (probability <= 1)
            saved = np.where(probability_valid, probability, -9999.0).astype("float32")
            probability_path = probability_dir / (row["case_id"] + "_probabilidad.tif")
            with rasterio.open(probability_path, "w", **data["profile"]) as output:
                output.write(saved, 1)
                output.set_band_description(1, "water_probability")
                output.update_tags(checkpoint_sha256=checkpoint_record["sha256"],
                                   reference_origin=REFERENCE_ORIGIN, scene_id=row["scene_id"],
                                   l1c_source=row["l1c_source"], l2a_product=row["l2a_product"])
            with rasterio.open(probability_path) as saved_probability:
                saved_values = saved_probability.read(1)
                saved_mask = saved_probability.read_masks(1) != 0
                require(saved_probability.transform == data["profile"]["transform"] and
                        saved_probability.crs == data["profile"]["crs"] and
                        saved_probability.shape == (256, 256), "La probabilidad guardada cambió de cuadrícula.")
            scores, control, _ = evaluate_arrays(data["optical"], data["reference"], data["validity"],
                                                saved_values, **data["masks"], probability_gdal=saved_mask)
            identifiers = {"case_id": row["case_id"], "scene_id": row["scene_id"], "cluster_id": row["cluster_id"]}
            for score in scores:
                per_case.append({**identifiers, **score})
            controls.append({**identifiers, **control, "input_pixels_zero_filled": int((~data["input_valid"]).sum())})
            probability_sources.append({**identifiers, "path": str(probability_path),
                                        "sha256": sha256(probability_path),
                                        "l1c_source": row["l1c_source"], "l2a_product": row["l2a_product"]})
        print(f"Inferencia: {min(start + batch_size, len(rows))}/{len(rows)} casos procesados.",
              file=sys.stderr, flush=True)
    require(sum(row["valid_pixels"] for row in per_case if row["method"] == "unet") > 0,
            "La cohorte final no tiene píxeles comunes; no se generan tablas de métricas vacías.")
    case_table = aggregate(per_case, "case", "case_id")
    scenes = aggregate(per_case, "scene", "scene_id")
    clusters = aggregate(per_case, "cluster", "cluster_id")
    global_rows = aggregate(per_case, "global")
    for name, table in (("metricas_por_caso.csv", per_case), ("metricas_por_escena.csv", scenes),
                        ("metricas_por_cluster.csv", clusters), ("metricas_globales.csv", global_rows),
                        ("diferencias_pareadas.csv", paired_differences([case_table, scenes, clusters, global_rows])),
                        ("controles_por_caso.csv", controls)):
        write_csv(out / name, table)
    require(sha256(manifest) == manifest_hash, "El manifiesto cambió durante la ejecución.")
    common_metadata.update(
        status="external_evaluation_completed", inference_executed=True, evaluation_executed=True,
        metric_outputs_created=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=checkpoint_record, probability_outputs=probability_sources,
        common_valid_pixels=global_rows[0]["valid_pixels"],
        aggregation="Pooled integer counts from distinct preselected windows; scene/cluster/global are separate views, never resummed",
        undefined_metric_convention="0/0 is stored as zero with metric_defined=false; undefined differences are empty",
        common_support="All five optical bands, reference and VALID GDAL masks, VALID=1, finite optics and probability, both denominators positive",
        evaluator_sha256=sha256(Path(__file__)),
    )
    if "tensorflow" in sys.modules:
        common_metadata["runtime"]["tensorflow"] = sys.modules["tensorflow"].__version__
    write_json(out / "evaluacion_metadatos.json", common_metadata)
    return common_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--plan", help="Plan congelado; por defecto plan_congelado.json junto al manifiesto.")
    args = parser.parse_args(argv)
    try:
        metadata = run(args.manifest, args.checkpoint, args.out_dir,
                       validate_only=args.validate_only, batch_size=args.batch_size, plan_path=args.plan)
    except (ValidationError, FileNotFoundError, RuntimeError, OSError) as error:
        parser.exit(2, "ERROR: " + str(error) + "\n")
    print(json.dumps({key: metadata[key] for key in ("status", "n_cases", "n_scenes", "n_clusters",
                                                   "inference_executed", "evaluation_executed")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
