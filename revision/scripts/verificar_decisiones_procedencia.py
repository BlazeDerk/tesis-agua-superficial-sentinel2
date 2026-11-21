"""Verificación documental local del estudio. Uso: python verificar_decisiones_procedencia.py RAIZ SALIDA"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, data):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def indexed(records, key):
    result = {r[key]: r for r in records}
    assert len(result) == len(records), f"Identificador duplicado: {key}"
    return result


def run(root, out):
    root, out = root.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    train = "entrenamiento/"
    smoke = train + "smoke_multirregional_20261003/"
    final = train + "final_multirregional_20261004/"
    audit = "auditoria_20260930/"
    prior = "ampliacion_piloto_20261117/procedencia/"
    diag = "diagnostico_20261005/"
    sources = {
        "D01": (train+"preflight_multirregional_completo_20261003.json", "checked_utc; pipeline, semilla, filtros, población y muestra anteriores al ensayo"),
        "D02": (train+"preflight_multirregional_completo_20261003_partition.csv", "asignación por ID, bloque y hashes antes de inferencia"),
        "D03": ("fase3_entrenar_unet.py", "pérdida, arquitectura, aumentos, selección, umbral y valores por defecto; SHA registrado en D01"),
        "D04": (smoke+"config.json", "parámetros de invocación del ensayo"),
        "D05": (smoke+"report.json", "configuración efectiva y ejecución del ensayo"),
        "D06": (smoke+"splits.csv", "40 participantes efectivos del ensayo"),
        "D07": (audit+"verificacion_entrega_final_20261003.json", "verificación de cuatro pruebas; revisión de figuras registrada"),
        "D08": (train+"lanzamiento_final_20261004.json", "nota de exposición, lanzamiento y parámetros"),
        "D09": (final+"config.json", "parámetros de invocación final"),
        "D10": (final+"report.json", "entrenamiento, selección por validación y evaluación final"),
        "D11": (final+"splits.csv", "561 participantes efectivos, bloque y hashes"),
        "D12": (final+"resume_context.json", "fecha, reinicio, semilla, estado de optimizador y regla de selección"),
        "D13": (final+"history.csv", "50 épocas, pérdidas de validación y tasa observada"),
        "D14": ("Paquete_para_revision_articulo/tablas/cuatro_recortes_test_expuestos.csv", "lista histórica de solapamiento"),
        "D15": (diag+"diagnostico_train_val.json", "evaluación diagnóstica posterior"),
        "D16": (diag+"LEEME_DIAGNOSTICO.txt", "19 umbrales en validación y mantenimiento del umbral operativo"),
        "D17": (diag+"ensayo_agua_pequena/report_experimental.json", "ensayo posterior y evidencia de conservación de archivos protegidos"),
        "D18": (diag+"ensayo_agua_pequena/config_experimental.json", "condiciones del ensayo posterior"),
        "D19": (smoke+"spatial_partition.csv", "población completa antes de seleccionar 40"),
        "D20": (final+"spatial_partition.csv", "población completa para corrida final"),
        "D21": (train+"instrucciones_smoke_multirregional.txt", "instrucciones técnicas distinguen explícitamente el ensayo de dos épocas"),
        "D22": (final+"model_best.keras", "checkpoint de la corrida evaluada"),
        "D23": (train+"interrupcion_final_20261005/interrupcion.json", "registro del reinicio y segmentos perdidos"),
        "D24": (train+"ajuste_rendimiento_final_20261004.json", "afinidad CPU por tiempos de época"),
        "D25": (train+"ajuste_prioridad_final_20261004.json", "prioridad del proceso por contención de CPU"),
        "D26": (train+"ajuste_highqos_reanudacion_final_20261005.json", "gestión del proceso reanudado"),
        "P01": ("Dataset_UNet/Contexto_Dataset_UNet_Tesis.md", "documento de traspaso del legado; autor del texto y método declarado"),
        "P02": (audit+"auditoria_pares.csv", "555 pares físicos del legado con etiquetas, metadatos y huellas"),
        "P03": (audit+"correcciones_verificadas.csv", "conservación de valores, georreferencia y huellas"),
        "P04": (audit+"resumen_auditoria.json", "tipos originales y resumen de anomalías"),
        "P05": ("Dataset_UNet_Unificado/manifest.csv", "relación origen–ventana–ubicación geográfica–huellas"),
        "P06": ("gee/fase1_ejecutado_navegador.js", "script histórico de los tres sectores; unión y filtros"),
        "P07": ("entradas_fase1/S2_UNet_Fase1_proveniencia.geojson", "metadatos exportados por zona"),
        "P08": ("entradas_fase1_corregidas/S2_UNet_Fase1_proveniencia.geojson", "copia conservada con los mosaicos corregidos"),
        "P09": ("gee/exportaciones_completadas_20261003.json", "siete tareas con ID y estado"),
        "P10": ("Paquete_para_revision_articulo/tablas/cronologia_gee.csv", "fechas de exportación y descarga; distintas de la adquisición satelital"),
        "P11": (prior+"hallazgos_procedencia.json", "auditoría previa de fuentes del proyecto"),
        "P12": (prior+"consulta_permiso_el_oro.txt", "borrador de consulta al titular"),
    }
    path = lambda k: root / sources[k][0]
    load = lambda k: json.loads(path(k).read_text(encoding="utf-8-sig"))
    checksums = {k:digest(path(k)) for k in sources}
    write_csv(out/"fuentes_sha256.csv", [
        dict(id=k, ruta_relativa_al_proyecto=v[0], bytes=path(k).stat().st_size,
             sha256=checksums[k], evidencia=v[1]) for k,v in sources.items()])
    pre, trial, official, launch, resume = (load(k) for k in ["D01","D05","D10","D08","D12"])
    fields = {
        "D01": ["checked_utc","pipeline_sha256","manifest_sha256","seed","seed_search_performed","seed_selected_from_test_metrics","spatial_block_m","buffer_m","model_constructed","test_evaluated","admitted_count","loader_excluded_counts_by_zone","buffer_excluded_count"],
        "D05": ["timestamp_utc","mode","split_counts","epochs_completed","trainable_parameters","pretrained_transfer","pipeline_sha256","manifest_sha256","test_evaluations"],
        "D07": ["checked_utc","actual_smoke_samples","actual_smoke_epochs","root_figures_visually_inspected"],
        "D08": ["launched_utc","parameters","pipeline_sha256","prior_smoke_test_exposure"],
        "D10": ["timestamp_utc","completion_timestamp_utc","mode","split_counts","epochs_completed","trainable_parameters","pretrained_transfer","best_validation_epoch_one_based","selected_checkpoint_sha256","test_evaluations"],
        "D12": ["timestamp_utc","interruption_reason","initial_epoch","next_epoch_one_based","total_epochs_requested","additional_epochs_maximum","prior_checkpoint_optimizer_iterations","seed_original","seed_training_start","rng_continuity_limit","test_evaluations_before_resume","known_previous_smoke_test_exposure","selection_rule"],
        "D17": ["timestamp_utc","sample_unique_count","sample_draw_count","test_samples_read","deployment_model_changed","protected_artifacts_unchanged","model_experimental_sha256","deployment_reason"],
        "D18": ["epochs","batch_size","seed","learning_rate","optimizer","loss","encoder_frozen","sample_size","val_evaluation","test_evaluation","deployment_model_changed"],
        "P07": ["type","features"],
        "P09": ["observed_utc","owned_tasks"]
    }
    extracts = {}
    for k, wanted in fields.items():
        original = load(k)
        extracts[k] = {"sha256_del_archivo_completo":checksums[k],
                       "campos_originales":{field:original[field] for field in wanted}}
    (out/"extractos_de_registros_originales.json").write_text(
        json.dumps(extracts,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    assert checksums["D03"] == pre["pipeline_sha256"] == trial["pipeline_sha256"] == official["pipeline_sha256"] == launch["pipeline_sha256"]
    assert checksums["P05"] == pre["manifest_sha256"] == trial["manifest_sha256"] == official["manifest_sha256"]
    assert checksums["D19"] == checksums["D20"]
    assert checksums["D22"] == official["selected_checkpoint_sha256"] == "9e28f92b557bdd2e9a2e7b068a6622254ddbf80080926b8c004368ad5683dda4"
    planned = indexed(rows(path("D02")), "name")
    observed = indexed(rows(path("D11")), "name")
    sampled = indexed(rows(path("D06")), "name")
    assert set(observed) == {n for n,r in planned.items() if r["split_after_buffer"]}
    assert set(sampled) == {n for n,r in planned.items() if r["selected_smoke40"] == "True"}
    for cohort in [observed, sampled]:
        for n,r in cohort.items():
            before = planned[n]
            for current, prev in [("split","split_after_buffer"),("spatial_group","spatial_group"),
                                  ("bounds","bounds"),("zone","zone"),("source_sha256","image_sha256"),("mask_sha256","mask_sha256")]:
                assert r[current] == before[prev], (n,current)
    exposure = {n for n, r in sampled.items() if r["split"] == "test"} & {n for n, r in observed.items() if r["split"] == "test"}
    assert len(exposure) == 4 and exposure == {r["final_name"] for r in rows(path("D14"))}
    write_csv(out/"cuatro_test_previamente_expuestos.csv", [
        dict(recorte=n, zona=observed[n]["zone"], bloque=observed[n]["spatial_group"],
             sha256_imagen=observed[n]["source_sha256"], sha256_mascara=observed[n]["mask_sha256"],
             rol_ensayo="test", rol_final="test") for n in sorted(exposure)])
    manifest = [r for r in rows(path("P05")) if r["kind"] == "legacy"]
    audited = indexed(rows(path("P02")), "name")
    repaired = indexed(rows(path("P03")), "name")
    assert len(manifest) == len(audited) == len(repaired) == 555
    assert len({PureWindowsPath(r["source_image"]).name for r in manifest}) == len(manifest)
    mapped = []
    for m in manifest:
        name = PureWindowsPath(m["source_image"]).name
        a, r = audited[name], repaired[name]
        img = digest(root/"Dataset_UNet/images"/name)
        mask = digest(root/"Dataset_UNet/masks"/name)
        unified_image = digest(root/"Dataset_UNet_Unificado"/m["image"])
        unified_mask = digest(root/"Dataset_UNet_Unificado"/m["mask"])
        assert img == a["image_sha256"] == m["output_image_sha256"] == unified_image
        assert mask == a["mask_sha256"] == r["source_mask_sha256"]
        assert unified_mask == r["corrected_mask_sha256"] == m["output_mask_sha256"]
        assert all(r[k]=="True" for k in ["mask_pixel_values_identical","mask_crs_identical","mask_transform_identical"])
        patch = Path(m["image"]).name
        mapped.append(dict(original=name, recorte_unificado=patch, zona=m["zone"],
                           rol_final=observed.get(patch,{}).get("split",""),
                           crs=m["crs"], limites_en_metros=m["bounds"],
                           sha256_imagen_original_y_unificada=img, sha256_mascara_original=mask,
                           sha256_mascara_unificada=unified_mask, fecha_adquisicion="",
                           id_granulo="", estado_relacion_con_escenas="sin registro individual"))
    write_csv(out/"trazabilidad_555_el_oro.csv",mapped)
    assert checksums["P07"] == checksums["P08"]
    # La lista de miembros se lee sin extraer ni modificar los originales.
    archive = Path.home()/"Downloads"/"Dataset_UNet.rar"
    archive_summary = {"estado": "archivo o lector TAR no disponible"}
    if archive.is_file() and shutil.which("tar"):
        listing = subprocess.run(["tar","-tf",str(archive)],check=True,capture_output=True,text=True)
        members = [Path(n.strip()) for n in listing.stdout.splitlines() if n.strip()]
        tiffs = [p for p in members if p.suffix.lower()==".tif"]
        assert {p.name for p in tiffs if p.parent.name.lower()=="images"} == set(audited)
        assert {p.name for p in tiffs if p.parent.name.lower()=="masks"} == set(audited)
        archive_summary = dict(nombre=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,
                               tif=len(tiffs), otros_miembros=[p.as_posix() for p in members if p not in tiffs])
    zones = []
    for entry in load("P07")["features"]:
        p = entry["properties"]
        zones.append(dict(sector=p["zone"], poligono_lonlat=json.dumps(entry["geometry"]["coordinates"],separators=(",",":")),
                          coleccion=p["sr_collection"], coleccion_nubes=p["cloud_collection"],
                          inicio_inclusivo=p["start"], fin_exclusivo=p["end_exclusive"],
                          pares_de_granulos_tras_union=p["matched_scene_count"],
                          nubosidad_escena_menor_que_pct=p["scene_cloud_percent_max"],
                          probabilidad_nube_desde_pct=p["cloud_probability_threshold"],
                          dilatacion_m=p["cloud_buffer_m"], minimo_observaciones_claras=p["min_clear_observations"],
                          PRODUCT_ID="", system_index="", adquisicion_utc="",
                          estado_ids="solo conteo por sector archivado; lista individual no recuperada"))
    write_csv(out/"procedencia_mosaicos_gee.csv",zones)
    history = rows(path("D13"))
    lr_changes = []
    for i,r in enumerate(history, start=1):
        learning_rate = float(r["learning_rate"])
        if not lr_changes or learning_rate != lr_changes[-1]["tasa"]:
            lr_changes.append(dict(primera_epoca_registrada=i,tasa=learning_rate))
    validation = dict(chequeado_utc=datetime.now(timezone.utc).isoformat(), metodo="comparación local de registros y SHA-256 de archivos actuales",
                      prueba_previa_utc=pre["checked_utc"], inicio_ensayo_utc=trial["timestamp_utc"],
                      verificacion_ensayo_utc=load("D07")["checked_utc"], lanzamiento_final_utc=launch["launched_utc"],
                      reanudacion_utc=resume["timestamp_utc"], termino_final_utc=official["completion_timestamp_utc"],
                      declaracion_original_lanzamiento=launch["prior_smoke_test_exposure"], parametros_entrenables_ensayo=trial["trainable_parameters"],
                      parametros_entrenables_final=official["trainable_parameters"], sha256_codigo=checksums["D03"],
                      sha256_manifest=checksums["P05"], sha256_checkpoint=checksums["D22"],
                      fase_previa_vs_final_561_coinciden=True, fase_previa_vs_ensayo_40_coinciden=True,
                      particion_completa_smoke_final_sha_identico=True, cuatro_expuestos=len(exposure),
                      reparto_final=dict(Counter(r["split"] for r in observed.values())),
                      epoca_seleccionada_por_val_loss=official["best_validation_epoch_one_based"], tasa_por_epoca=lr_changes,
                      originales_releidos_y_verificados=1110, unificados_del_legado_releidos_y_verificados=1110,
                      inventario_archivo_de_traspaso=archive_summary,
                      distribucion_legado={k or "fuera del split":v for k,v in Counter(r["rol_final"] for r in mapped).items()},
                      tareas_gee_documentadas=len(load("P09")["owned_tasks"]), catalogo_gee_reconsultado=False)
    (out/"verificacion_documental.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raiz",type=Path)
    parser.add_argument("salida",type=Path)
    args = parser.parse_args()
    run(args.raiz,args.salida)
