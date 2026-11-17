"""Baseline NDWI fijo sobre los 59 Test congelados, sin ejecutar la U-Net.

Uso:
python baseline_ndwi.py --splits splits.csv --exposure expuestos.csv \
    --dataset-root Dataset_UNet_Unificado --out-dir resultados_ndwi

Solo lee TIFF originales. Falla ante hashes ausentes/discrepantes, exposición
desconocida, geometrías incompatibles o diferencias de validez con el Test
original (59 x 256 x 256 píxeles válidos). No ajusta umbrales con Test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import numpy as np
    import rasterio
except ImportError as exc:
    raise SystemExit("Se requieren NumPy y Rasterio: python -m pip install numpy rasterio") from exc

ZONES = ("el_oro", "golfo_guayaquil", "lagos_imbabura", "rio_napo")
THRESHOLD = 0.0
EXPECTED_TEST_PATCHES = 59
EXPECTED_PRIOR_EXPOSURES = 4
SHAPE = (256, 256)
TOTAL_ORIGINAL_VALID_PIXELS = EXPECTED_TEST_PATCHES * SHAPE[0] * SHAPE[1]


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def file_name(value: str) -> str:
    # Permite inspeccionar manifiestos Windows también en otros sistemas.
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def require_sha(value: str | None, label: str) -> str:
    if not value or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"Falta SHA-256 válido: {label}")
    return value.lower()


def verified_hash(path: Path, expected_values: list[tuple[str, str | None]]) -> str:
    expected = [require_sha(v, f"{path.name}: {label}") for label, v in expected_values]
    if len(set(expected)) != 1:
        raise ValueError(f"Los manifiestos discrepan para {path.name}")
    actual = sha256(path)
    if actual != expected[0]:
        raise ValueError(f"SHA-256 discrepante para {path.name}")
    return actual


def index_manifest(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        name = file_name(row.get("image", ""))
        if not name or name in result:
            raise ValueError(f"Imagen vacía o duplicada en manifiesto: {name!r}")
        result[name] = row
    return result


def validate_exposures(rows: list[dict[str, str]], tests: list[dict[str, str]], expected_count=4) -> set[str]:
    tests_by_name = {r["name"]: r for r in tests}
    if len(tests_by_name) != len(tests):
        raise ValueError("Recortes Test duplicados")
    found = set()
    for row in rows:
        name = row.get("final_name", "")
        if name not in tests_by_name:
            raise ValueError(f"Exposición no pertenece al Test congelado: {name!r}")
        if name in found:
            raise ValueError(f"Exposición duplicada: {name}")
        if row.get("final_split") != "test" or row.get("zone") != tests_by_name[name]["zone"]:
            raise ValueError(f"Exposición con partición/zona incompatible: {name}")
        if row.get("prior_split") != "test" or not row.get("prior_experiment"):
            raise ValueError(f"Procedencia incompleta de exposición: {name}")
        found.add(name)
    if len(found) != expected_count:
        raise ValueError(f"Se esperaban {expected_count} exposiciones, se obtuvieron {len(found)}")
    return found


def ndwi_arrays(green, nir, valid_mask):
    """Devuelve predicción >0 y diagnósticos, sin excluir píxeles en silencio."""
    green = np.asarray(green, dtype=np.float64)
    nir = np.asarray(nir, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if green.shape != nir.shape or green.shape != valid_mask.shape:
        raise ValueError("Forma incompatible al calcular NDWI")
    denominator = green + nir
    finite = np.isfinite(green) & np.isfinite(nir) & np.isfinite(denominator)
    defined = finite & (denominator > 0)
    index = np.full(green.shape, np.nan, dtype=np.float64)
    np.divide(green - nir, denominator, out=index, where=defined)
    prediction = defined & (index > THRESHOLD)
    diagnostics = {
        "denominator_nonpositive_on_original_valid": int(np.count_nonzero(valid_mask & finite & (denominator <= 0))),
        "nonfinite_on_original_valid": int(np.count_nonzero(valid_mask & ~finite)),
        "ndwi_undefined_on_original_valid": int(np.count_nonzero(valid_mask & ~defined)),
        "ndwi_equal_threshold_on_original_valid": int(np.count_nonzero(valid_mask & defined & (index == THRESHOLD))),
        "ndwi_min_on_original_valid": float(np.nanmin(index[valid_mask & defined])) if np.any(valid_mask & defined) else None,
        "ndwi_max_on_original_valid": float(np.nanmax(index[valid_mask & defined])) if np.any(valid_mask & defined) else None,
    }
    return prediction, diagnostics


def ensure_original_cohort(valid_mask, ndwi_diagnostics):
    if not np.all(valid_mask):
        raise ValueError("Validez distinta del Test original: no se excluirán píxeles silenciosamente")
    if ndwi_diagnostics["ndwi_undefined_on_original_valid"]:
        raise ValueError("NDWI indefinido en píxeles del Test original; comparación pareada detenida")


def confusion(reference, prediction, valid):
    reference, prediction, valid = (np.asarray(x, dtype=bool) for x in (reference, prediction, valid))
    if not (reference.shape == prediction.shape == valid.shape):
        raise ValueError("Forma incompatible en conteos")
    tp = int(np.count_nonzero(valid & reference & prediction))
    fp = int(np.count_nonzero(valid & ~reference & prediction))
    fn = int(np.count_nonzero(valid & reference & ~prediction))
    tn = int(np.count_nonzero(valid & ~reference & ~prediction))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "valid_pixels": tp + fp + fn + tn, "water_reference_pixels": tp + fn,
            "water_predicted_pixels": tp + fp}


def metric_values(counts):
    tp, fp, fn = (int(counts[k]) for k in ("tp", "fp", "fn"))
    denominators = {"iou": tp + fp + fn, "f1": 2 * tp + fp + fn,
                    "precision": tp + fp, "recall": tp + fn}
    numerators = {"iou": tp, "f1": 2 * tp, "precision": tp, "recall": tp}
    result = {}
    for name, den in denominators.items():
        result[name] = numerators[name] / den if den else 0.0
        result[name + "_defined"] = bool(den)
    return result


def summarize(patch_rows, exposed):
    result = []
    cohorts = [("all_test", patch_rows),
               ("without_previous_exposure", [r for r in patch_rows if r["name"] not in exposed]),
               ("previous_exposure", [r for r in patch_rows if r["name"] in exposed])]
    count_columns = ("valid_pixels", "water_reference_pixels", "water_predicted_pixels", "tp", "fp", "fn", "tn")
    for cohort, items in cohorts:
        zone_rows = []
        for zone in ("ALL", *ZONES):
            subset = items if zone == "ALL" else [r for r in items if r["zone"] == zone]
            counts = {k: sum(int(r[k]) for r in subset) for k in count_columns}
            record = {"cohort": cohort, "aggregation": "global_pixels" if zone == "ALL" else "zone",
                      "zone": zone, "n_patches": len(subset), **counts,
                      "n_zones_present": len({r["zone"] for r in subset}), "n_zones_expected": 4,
                      "missing_zones": ";".join(z for z in ZONES if not any(r["zone"] == z for r in subset)),
                      "status": "evaluated" if subset else "no_samples", "note": ""}
            if subset:
                record.update(metric_values(counts))
            else:
                record.update({k: None for k in ("iou", "f1", "precision", "recall")})
                record.update({k + "_defined": False for k in ("iou", "f1", "precision", "recall")})
            result.append(record)
            if zone != "ALL" and subset:
                zone_rows.append(record)
        macro = {"cohort": cohort, "aggregation": "unweighted_region_mean", "zone": "PRESENT_ZONES",
                 "n_patches": len(items), "n_zones_present": len(zone_rows), "n_zones_expected": 4,
                 "missing_zones": ";".join(z for z in ZONES if not any(r["zone"] == z for r in items)),
                 "status": "evaluated" if zone_rows else "no_samples",
                 "note": "Mean of ratios of present zones; confusion counts are not averaged. A missing zone is not assigned zero."}
        macro.update({k: None for k in count_columns})
        for key in ("iou", "f1", "precision", "recall"):
            macro[key] = sum(r[key] for r in zone_rows) / len(zone_rows) if zone_rows else None
            macro[key + "_defined"] = all(r[key + "_defined"] for r in zone_rows) if zone_rows else False
        result.append(macro)
    return result


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    start = datetime.now(timezone.utc)
    dataset = Path(args.dataset_root).resolve()
    split_path, exposure_path = Path(args.splits).resolve(), Path(args.exposure).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else dataset / "manifest.csv"
    split_rows = rows_csv(split_path)
    tests = [r for r in split_rows if r.get("split") == "test"]
    if len(tests) != EXPECTED_TEST_PATCHES:
        raise ValueError(f"Se esperaban 59 Test congelados, se obtuvieron {len(tests)}")
    if {r["zone"] for r in tests} != set(ZONES):
        raise ValueError("Las cuatro zonas originales no coinciden")
    manifest = index_manifest(rows_csv(manifest_path))
    exposed = validate_exposures(rows_csv(exposure_path), tests, EXPECTED_PRIOR_EXPOSURES)
    patch_rows, source_assets = [], []
    for row in tests:
        name = row["name"]
        if name != file_name(name):
            raise ValueError(f"Nombre de recorte no simple: {name}")
        if name not in manifest:
            raise ValueError(f"Recorte Test sin registro en el manifiesto: {name}")
        entry = manifest[name]
        if entry.get("zone") != row["zone"] or file_name(entry.get("mask", "")) != name:
            raise ValueError(f"Zona/pareja imagen-máscara incompatibles: {name}")
        image_path, reference_path = dataset / "images" / name, dataset / "masks" / name
        image_hash = verified_hash(image_path, [("manifest.output_image_sha256", entry.get("output_image_sha256")),
                                                ("splits.source_sha256", row.get("source_sha256"))])
        reference_hash = verified_hash(reference_path, [("manifest.output_mask_sha256", entry.get("output_mask_sha256")),
                                                        ("splits.mask_sha256", row.get("mask_sha256"))])
        with rasterio.open(image_path) as image, rasterio.open(reference_path) as reference:
            if image.count != 4 or (image.height, image.width) != SHAPE or reference.count != 1 or (reference.height, reference.width) != SHAPE:
                raise ValueError(f"Dimensiones distintas de 4x256x256 y máscara: {name}")
            if set(image.dtypes) != {"uint16"} or reference.dtypes != ("uint16",):
                raise ValueError(f"Tipos originales uint16 distintos: {name}")
            if image.crs != reference.crs or image.crs.to_epsg() != 32717:
                raise ValueError(f"CRS incompatible: {name}")
            if not image.transform.almost_equals(reference.transform, precision=1e-9):
                raise ValueError(f"Imagen y referencia no alineadas: {name}")
            a, b, c, d, e, f = tuple(image.transform)[:6]
            if not (a == 10 and b == 0 and d == 0 and e == -10):
                raise ValueError(f"Cuadrícula original 10 m incompatible: {name}")
            raw, labels = image.read(), reference.read(1)
            image_valid = np.all(image.read_masks() > 0, axis=0)
            reference_valid = reference.read_masks(1) > 0
            sentinel = np.any(raw == 65535, axis=0) | (labels == 65535)
            four_zero = np.all(raw == 0, axis=0)
            valid = image_valid & reference_valid & ~sentinel
            if np.any(four_zero):
                raise ValueError(f"Ceros en cuatro canales no presentes en Test auditado: {name}")
            if not np.all(np.isin(labels[valid], [0, 1])):
                raise ValueError(f"Referencia no binaria: {name}")
            predicted, diagnostics = ndwi_arrays(raw[1], raw[3], valid)
            ensure_original_cohort(valid, diagnostics)
            counts = confusion(labels == 1, predicted, valid)
            bounds = list(image.bounds)
            descriptions = list(image.descriptions)
        result = {"name": name, "zone": row["zone"], "split": "test",
                  "previous_exposure": name in exposed, "spatial_group": row["spatial_group"],
                  "threshold": THRESHOLD, "threshold_operator": ">", **counts, **metric_values(counts),
                  **diagnostics, "nodata_pixels": int(np.count_nonzero(~(image_valid & reference_valid))),
                  "sentinel_65535_pixels": int(np.count_nonzero(sentinel)), "four_channels_zero_pixels": int(np.count_nonzero(four_zero)),
                  "image_sha256": image_hash, "reference_sha256": reference_hash,
                  "manifest_hashes_verified": True, "frozen_split_hashes_verified": True,
                  "band_order_evidence": entry.get("band_order_evidence", ""),
                  "label_type_in_manifest": entry.get("label_type", ""),
                  "label_provenance_scope": "legacy_origin_not_fully_reconstructed" if row["zone"] == "el_oro" else "automatic_MNDWI_reference",
                  "image_path": f"images/{name}", "reference_path": f"masks/{name}",
                  "bounds_epsg32717": json.dumps(bounds), "embedded_band_descriptions": json.dumps(descriptions)}
        patch_rows.append(result)
        source_assets.append({"name": name, "image": f"images/{name}", "image_sha256": image_hash,
                              "reference": f"masks/{name}", "reference_sha256": reference_hash})
    if sum(r["valid_pixels"] for r in patch_rows) != TOTAL_ORIGINAL_VALID_PIXELS:
        raise ValueError("Total de píxeles distinto del Test original")
    if len(patch_rows) != 59 or sum(not r["previous_exposure"] for r in patch_rows) != 55:
        raise ValueError("Cohortes 59/55 incompatibles con los registros")
    summaries = summarize(patch_rows, exposed)
    end = datetime.now(timezone.utc)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("ndwi_por_recorte.csv", "ndwi_resumen.csv", "ndwi_metadatos.json")) and not args.overwrite:
        raise FileExistsError("Las salidas ya existen; elija otra carpeta o use --overwrite explícitamente")
    write_csv(out / "ndwi_por_recorte.csv", patch_rows)
    write_csv(out / "ndwi_resumen.csv", summaries)
    source_paths = {"splits": split_path, "exposure": exposure_path, "manifest": manifest_path,
                    "script": Path(__file__).resolve()}
    metadata = {
        "experiment_kind": "fixed_NDWI_baseline_new_analysis_of_frozen_original_Test",
        "started_at_utc": start.isoformat(), "completed_at_utc": end.isoformat(),
        "started_at_America_Guayaquil": start.astimezone(timezone(timedelta(hours=-5))).isoformat(),
        "completed_at_America_Guayaquil": end.astimezone(timezone(timedelta(hours=-5))).isoformat(),
        "elapsed_seconds": (end-start).total_seconds(),
        "date_scope": "This timestamp describes this baseline execution, not original acquisition or U-Net training.",
        "overwrite": bool(args.overwrite),
        "runtime": {"python": sys.version, "numpy": np.__version__, "rasterio": rasterio.__version__,
                    "gdal": rasterio.__gdal_version__, "platform": platform.platform()},
        "formula": "NDWI = (B3 - B8) / (B3 + B8)", "calculation_dtype": "float64",
        "input_values": "Original uint16 digital numbers cast to float64 before subtraction; a common division by 10000 cancels in this ratio.",
        "bands": {"B3_green_1based": 2, "B8_NIR_1based": 4},
        "threshold": THRESHOLD, "threshold_operator": ">", "threshold_selection": "fixed a priori in request; no validation or Test tuning",
        "original_U_Net_rerun": False, "training_performed": False, "original_files_modified": False,
        "source_files": {role: {"filename": path.name, "sha256": sha256(path)} for role,path in source_paths.items()},
        "asset_hash_validation": "Every image/reference matched both output hashes in dataset manifest and corresponding hashes in frozen splits.",
        "sources": source_assets,
        "test_patches": len(patch_rows), "original_valid_pixels": TOTAL_ORIGINAL_VALID_PIXELS,
        "cohorts": {"all_test": 59, "without_previous_exposure": 55, "previous_exposure": 4},
        "exposure_names": sorted(exposed),
        "validity_policy": "GDAL masks for all four input bands and reference, no uint16 sentinel65535; strict failure on any invalid or undefined NDWI pixel in originally all-valid Test. No silent pixel exclusion or changed denominators.",
        "all59_original_pixel_support_verified": True,
        "ndwi_undefined_on_original_valid": sum(r["ndwi_undefined_on_original_valid"] for r in patch_rows),
        "nodata_pixels": sum(r["nodata_pixels"] for r in patch_rows),
        "metric_formulas": {"iou": "TP/(TP+FP+FN)", "f1": "2*TP/(2*TP+FP+FN)",
                            "precision": "TP/(TP+FP)", "recall": "TP/(TP+FN)"},
        "metric_numeric_precision": "Ratios computed as Python float64 from exact int pixel counts; CSV preserves repr precision, no percentage rounding.",
        "zero_denominator_rule": "0.0 if denominator is zero (same numerical convention as original model), *_defined=False. No-sample zones instead have blank ratios/status=no_samples.",
        "aggregation": "Global and per-zone ratios from summed counts. Macro is unweighted mean of present-zone ratios. No-exposure cohort has no Imbabura samples, so its macro spans 3 zones, not 4.",
        "comparison_scope": "Concordance with original references. New references are automatic MNDWI labels; El Oro provenance is incomplete. Not independent field accuracy and not proof of superiority over real water.",
        "limitations": [
            "Legacy B2/B3/B4/B8 order is asserted by source context and not embedded in original TIFF band metadata.",
            "New mosaics are shared across spatial partitions; removing four prior exposures does not make the remaining55 an independent scene/year holdout.",
            "No threshold was tuned and no original model decision was changed.",
            "MNDWI evaluated against labels produced from MNDWI is only a consistency check without independent reference.",
        ],
        "output_files": {name: {"sha256": sha256(out/name), "bytes": (out/name).stat().st_size}
                         for name in ("ndwi_por_recorte.csv", "ndwi_resumen.csv")},
    }
    (out / "ndwi_metadatos.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return {"output_dir": str(out), "test_patches": len(patch_rows), "valid_pixels": TOTAL_ORIGINAL_VALID_PIXELS,
            "summaries": [{k:r[k] for k in ("cohort", "aggregation", "zone", "n_patches", "iou", "f1", "n_zones_present")}
                          for r in summaries if r["aggregation"] != "zone"]}


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--splits", required=True, type=Path)
    result.add_argument("--exposure", required=True, type=Path)
    result.add_argument("--dataset-root", required=True, type=Path)
    result.add_argument("--out-dir", required=True, type=Path)
    result.add_argument("--manifest", type=Path, help="Por defecto dataset-root/manifest.csv")
    result.add_argument("--overwrite", action="store_true", help="Permite reemplazar únicamente salidas derivadas")
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(run(parser().parse_args()), ensure_ascii=False, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError, rasterio.errors.RasterioError) as exc:
        raise SystemExit(f"Baseline detenido: {exc}") from exc
