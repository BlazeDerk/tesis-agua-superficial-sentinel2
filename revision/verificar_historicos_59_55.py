"""Comprueba desde los GeoTIFF los conteos históricos y genera las diferencias U-Net menos NDWI."""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio


METHODS = ("unet", "ndwi")
COUNTS = ("tp", "fp", "fn", "tn")
METRICS = ("iou", "f1", "precision", "recall")
ZONES = ("el_oro", "golfo_guayaquil", "lagos_imbabura", "rio_napo")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scores(c):
    tp, fp, fn, _ = map(int, c)
    pairs = [(tp, tp + fp + fn), (2*tp, 2*tp + fp + fn), (tp, tp + fp), (tp, tp + fn)]
    return {name: a / b if b else None for name, (a, b) in zip(METRICS, pairs)}


def check_table(row, counts, has_cases=True):
    assert int(row["valid_pixels"]) == sum(counts)
    assert int(row["water_reference_pixels"]) == counts[0] + counts[2]
    assert all(int(row[k]) == v for k, v in zip(COUNTS, counts))
    for metric, expected in scores(counts).items():
        assert (row[metric + "_defined"].lower() == "true") == (expected is not None)
        if not has_cases:
            assert row[metric] == "" and row["status"] == "no_samples"
        else:
            assert math.isclose(float(row[metric]), expected or 0, abs_tol=1e-14, rel_tol=0)


def delta(scope, cohort, identifier, zone, size, both):
    assert sum(both["unet"]) == sum(both["ndwi"])
    row = {"scope": scope, "cohort": cohort, "id": identifier, "zone": zone,
           "n_patches": size,"valid_pixels": int(sum(both["unet"])),
           "status": "evaluated" if size else "no_samples"}
    lhs, rhs = scores(both["unet"]), scores(both["ndwi"])
    for metric in METRICS:
        defined = lhs[metric] is not None and rhs[metric] is not None
        row[f"{metric}_unet_minus_ndwi_pp"] = 100 * (lhs[metric] - rhs[metric]) if defined else ""
        row[f"{metric}_difference_defined"] = defined
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["results","dataset-root","splits","exposure","original-report","original-predictions"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    out = args.results
    md = json.loads((out/"reevaluacion_historica_metadatos.json").read_text(encoding="utf-8-sig"))
    assert md["status"] == "historical_re_evaluation_completed"
    assert not md["threshold_tuning"] and not md["training_executed"]
    assert digest(args.splits) == md["inputs"]["source_files"]["splits"]["sha256"]
    assert digest(args.exposure) == md["inputs"]["source_files"]["exposure"]["sha256"]
    assert digest(args.original_report) == md["original_report"]["sha256"]
    historical = json.loads(args.original_report.read_text(encoding="utf-8-sig"))
    split = {r["name"]: r for r in read_csv(args.splits) if r["split"] == "test"}
    exposed = {r["final_name"] for r in read_csv(args.exposure)}
    probabilities = {r["name"]: r for r in md["probabilities"]}
    cases = read_csv(out/"historicos_por_recorte.csv")
    case_lookup = {(r["name"],r["method"]): r for r in cases}
    summaries = read_csv(out/"historicos_resumen.csv")
    lookup = {(r["cohort"],r["zone"],r["method"]): r for r in summaries}
    assert len(split) == len(probabilities) == len(list((out/"probabilidades").glob("*.tif"))) == 59
    assert set(split) == set(probabilities) and len(exposed) == 4 and exposed <= set(split)
    assert len(case_lookup) == len(cases) == 118 and len(summaries) == len(lookup) == 20
    counts, differences, archived = {}, [], []
    for name, info in split.items():
        prob = out / "probabilidades" / (Path(name).stem + "_probabilidad.tif")
        image, reference = args.dataset_root/"images"/name, args.dataset_root/"masks"/name
        assert digest(image) == info["source_sha256"] and digest(reference) == info["mask_sha256"]
        assert digest(prob) == probabilities[name]["sha256"]
        with rasterio.open(image) as s, rasterio.open(reference) as m, rasterio.open(prob) as p:
            assert s.shape == m.shape == p.shape == (256,256)
            assert s.transform == m.transform == p.transform and s.crs == m.crs == p.crs
            assert s.crs.to_epsg() == 32717 and p.dtypes == ("float32",)
            assert (s.read_masks() > 0).all() and (m.read_masks(1)>0).all() and (p.read_masks(1)>0).all()
            green, nir = s.read([2,4]).astype("int32")
            truth, prob_array = m.read(1).astype("uint8"), p.read(1)
            assert np.isin(truth, (0,1)).all() and ((green+nir)>0).all()
            assert np.isfinite(prob_array).all() and ((prob_array>=0)&(prob_array<=1)).all()
        pred = {"unet":prob_array >= .5, "ndwi": green > nir}
        both = {}
        for method in METHODS:
            tn, fp, fn, tp = np.bincount((2*truth + pred[method]).flatten(), minlength=4)
            c = counts[(name,method)] = both[method] = [int(tp),int(fp),int(fn),int(tn)]
            row = case_lookup[(name,method)]
            assert row["zone"] == info["zone"] and row["spatial_group"] == info["spatial_group"]
            assert (row["previous_exposure"].lower() == "true") == (name in exposed)
            check_table(row,c)
        differences.append(delta("case","all_test",name,info["zone"],1,both))
        old = args.original_predictions/(Path(name).stem + "_probability.tif")
        if old.is_file():
            with rasterio.open(old) as source:
                prior = source.read(1)
            archived.append({"name":name, "sha256_archived_raster":digest(old),
                             "max_abs_probability_difference":float(np.max(np.abs(prob_array - prior))),
                             "pixels_classified_differently":int(np.count_nonzero((prior >= .5) != pred["unet"]))})
    assert len(archived) == 4
    groups, comparison = {}, []
    for cohort, selection in [("all_test",set(split)),("without_previous_exposure",set(split)-exposed)]:
        assert len(selection) == (59 if cohort=="all_test" else 55)
        groups[cohort] = len({split[k]["spatial_group"] for k in selection})
        for zone in ["ALL", *ZONES]:
            names = [k for k in selection if zone=="ALL" or split[k]["zone"]==zone]
            both = {}
            for method in METHODS:
                c = both[method] = [sum(counts[k,method][i] for k in names) for i in range(4)]
                row = lookup[cohort,zone,method]
                assert int(row["n_patches"]) == len(names)
                check_table(row,c,bool(names))
            differences.append(delta("global_pixels" if zone=="ALL" else "zone",cohort,zone,zone,len(names),both))
            if cohort=="all_test":
                ref = historical["test_confusion"] if zone=="ALL" else historical["test_by_zone"][zone]
                comparison.append({"zone":zone,"new_minus_october":{k:v-ref[k] for k,v in zip(COUNTS,both["unet"])}})
    assert len(differences) == 69
    with (out/"diferencias_pareadas_historicas.csv").open("x",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(differences[0]))
        writer.writeheader()
        writer.writerows(differences)
    report = {"fecha_utc":datetime.now(timezone.utc).isoformat(),"estado":"verificado",
              "comparacion":"U-Net menos NDWI sobre los mismos píxeles; diferencias en puntos porcentuales",
              "casos":len(split),"conteos_y_metricas_por_caso_cotejados":len(cases),
              "conteos_y_metricas_agregados_cotejados":len(summaries),
              "diferencias_por_caso_y_agregadas":len(differences),"grupos_espaciales_en_cada_cohorte":groups,
              "sha256_codigo_verificacion":digest(Path(__file__)),
              "comparacion_octubre_regional_y_global":comparison,"comparacion_cuatro_rasteres_de_octubre":archived}
    with (out/"verificacion_independiente_historica.json").open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report, ensure_ascii=False))

if __name__=="__main__":
    main()
