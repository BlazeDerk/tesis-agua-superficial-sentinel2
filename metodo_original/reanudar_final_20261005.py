"""Continua la corrida FINAL04 desde su checkpoint completo de la epoca16.

No reconstruye el modelo, no vuelve a inicializar VGG/Adam y no modifica el
pipeline congelado. Conserva las16 filas existentes y limita la corrida a50
epocas TOTALES. El Test se usa una sola vez al finalizar la seleccion por Val.
Este helper se ejecuta solo despues de revisar el archivo de la interrupcion.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "entrenamiento" / "final_multirregional_20261004"
ARCHIVE = ROOT / "entrenamiento" / "interrupcion_final_20261005"
SOURCE = ROOT / "fase3_entrenar_unet.py"
DATASET = ROOT / "Dataset_UNet_Unificado"
INITIAL_EPOCH = 16
TOTAL_EPOCHS = 50
SOURCE_SHA = "a6e0c1a5fe44aa14fc69bc28772ad6a228f25b6d65272f2875f66cb4c83f405f"
HISTORY_SHA = "126543cdbb93a8a2ba07eb84f9ac667702566805a8fb09fb3ba0007ec9a8dcea"
CHECKPOINT_SHA = "be24d302d3ce1a322b8e9bd719bf8c777893217aff87dd5c2f7f5a4566c0d715"
MANIFEST_SHA = "b19c606660a27bc4e4c9ae1e4bad02f6074f1ed478afa2841a1480fe5fa6b7ba"
SUMMARY_SHA = "7799e9e189c032448b35c697b9eb4954755b32d1a78ff31521652d311f154080"
SPLITS_SHA = "1f116206fa6c55371284e01c84be3747dec59a262200f424e8e44df14edf49ce"
EXPECTED_COUNTS = {"train": 460, "val": 42, "test": 59}
EXPECTED_CONFIG = {
    "mode": "final", "epochs": 50, "batch_size": 4, "threads": 8,
    "seed": 42, "buffer_m": 2560.0, "learning_rate": 1e-4,
    "freeze_encoder": False, "allow_ambiguous_zeros": False,
    "smoke_samples": None,
}
VGG_CONVS = [f"block{block}_conv{conv}" for block, count in
             ((1, 2), (2, 2), (3, 3), (4, 3), (5, 3))
             for conv in range(1, count + 1)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp_{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def workspace_path(value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    require(path.is_relative_to(ROOT), "Ruta fuera del workspace autorizado.")
    return path


def frozen_guards():
    require(ARCHIVE.is_dir() and (ARCHIVE / "interrupcion.json").is_file(),
            "Falta el archivo historico de la interrupcion; no se escribe ni se entrena.")
    interruption = read_json(ARCHIVE / "interrupcion.json")
    require(interruption["completed_epochs"] == INITIAL_EPOCH and
            interruption["interrupted_during_epoch"] == 17 and
            interruption["test_evaluations_observed"] == 0 and
            interruption["original_process_active"] is False,
            "La interrupcion no acredita16epocas guardadas sin Test.")
    for name, expected in interruption["files_sha256"].items():
        require(sha256(ARCHIVE / name) == expected,
                f"Archivo historico alterado: {name}")
        require(sha256(RUN / name) == expected,
                f"Archivo activo cambio tras archivar: {name}; revisar antes de reanudar.")
    require(sha256(SOURCE) == SOURCE_SHA == interruption["pipeline_sha256"],
            "Fuente original alterada.")
    for path, expected in ((RUN / "history.csv", HISTORY_SHA),
                           (RUN / "model_best.keras", CHECKPOINT_SHA),
                           (RUN / "splits.csv", SPLITS_SHA),
                           (RUN / "spatial_partition.csv", SPLITS_SHA),
                           (DATASET / "manifest.csv", MANIFEST_SHA),
                           (DATASET / "summary.json", SUMMARY_SHA)):
        require(sha256(path) == expected, f"Huella congelada incorrecta: {path.name}")
    config = read_json(RUN / "config.json")
    require(all(config.get(key) == value for key, value in EXPECTED_CONFIG.items()),
            "Configuracion difiere del entrenamiento autorizado.")
    require(workspace_path(config["dataset"]) == DATASET and
            workspace_path(config["manifest"]) == DATASET / "manifest.csv" and
            workspace_path(config["output"]) == RUN, "Rutas de la corrida incorrectas.")
    report = read_json(RUN / "report.json")
    require(report["training_status"] == "started" and report["mode"] == "final" and
            report["pipeline_sha256"] == SOURCE_SHA and
            report["samples_used"] == 561 and report["split_counts"] == EXPECTED_COUNTS and
            report.get("test_evaluations", 0) == 0 and "test_metrics" not in report,
            "Reporte no permite esta recuperacion sin repetir Test.")
    require(not any((RUN / name).exists() for name in
                    ("model_final.keras", "predictions", "predictions.png",
                     "test_probability_cache.npz", "resume_context.json")),
            "Hay resultados finales o una reanudacion previa; no se repite ni sobrescribe.")
    history = read_csv(RUN / "history.csv")
    require(len(history) == INITIAL_EPOCH and
            [int(row["epoch"]) for row in history] == list(range(INITIAL_EPOCH)),
            "Historial no contiene exactamente0..15.")
    require(all(math.isfinite(float(value)) for row in history for value in row.values()),
            "Historial inicial contieneNaN/Inf.")
    losses = [float(row["val_loss"]) for row in history]
    require(losses[-1] == min(losses) == 0.2227257937192917,
            "Ultima epoca no es el mejor checkpoint16 esperado.")
    require(all(math.isclose(float(row["learning_rate"]), 1e-4, rel_tol=1e-7)
                for row in history), "Historia contiene una reduccionLR que requiere recuperar otro estado.")
    return config, report, history, interruption


def load_pipeline():
    specification = importlib.util.spec_from_file_location("tesis_pipeline_congelado", SOURCE)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def confirm_splits(pipeline, args, expected_rows, report):
    records, issues, manifest = pipeline.load_records(args)
    splits, discarded, population = pipeline.split_geographic(
        records, args.seed, args.buffer_m, None, stratify_by_zone=True, return_population=True)
    require(len(records) == 686 and len(discarded) == 125 and
            {key: len(rows) for key, rows in splits.items()} == EXPECTED_COUNTS,
            "Datos/buffer/particiones ya no son686/125/460-42-59.")
    actual = [row for split in ("train", "val", "test") for row in splits[split]]
    require(len(actual) == len(expected_rows) == 561, "Poblacion retenida incorrecta.")
    keys = ("name", "image_path", "mask_path", "source_image", "zone", "label_type",
            "split", "spatial_group", "source_sha256", "mask_sha256")
    for row, saved in zip(actual, expected_rows):
        require(all(str(row[key]) == saved[key] for key in keys) and
                row["bounds"] == json.loads(saved["bounds"]),
                f"Orden/pertenencia/procedencia difiere del split original: {row['name']}")
    require(issues == report["data_issues"] and manifest == report["manifest"] and
            {key: dict(Counter(row["zone"] for row in rows)) for key, rows in splits.items()}
            == report["split_region_counts"], "Procedencia/auditoria de datos cambio.")
    return splits


def confirm_loaded_checkpoint(pipeline, model):
    import h5py
    np = pipeline.np
    require(list(model.input_shape) == [None, 256, 256, 4] and
            list(model.output_shape) == [None, 256, 256, 1] and
            model.count_params() == 16588945 and len(model.trainable_variables) == 48 and
            sum(int(np.prod(variable.shape)) for variable in model.trainable_variables) == 16588945,
            "Modelo cargado no es laU-Net4canales totalmente entrenable.")
    encoder = model.get_layer("vgg16_encoder4")
    require([layer.name for layer in encoder.layers if isinstance(layer, pipeline.keras.layers.Conv2D)]
            == VGG_CONVS and all(encoder.get_layer(name).trainable for name in VGG_CONVS),
            "EncoderVGG16 no contiene las13convoluciones entrenables.")
    require(tuple(encoder.get_layer("block1_conv1").kernel.shape) == (3, 3, 4, 64) and
            model.get_layer("water_probability").activation.__name__ == "sigmoid",
            "Primera convolucion4canales/salidaSigmoid incorrectas.")
    require(not any(isinstance(layer, (pipeline.keras.layers.Dropout,
                                     pipeline.keras.layers.BatchNormalization))
                    for layer in list(model.layers) + list(encoder.layers)),
            "Modelo con estado estocastico/BN adicional: recuperacion requiere otra revision.")
    optimizer = model.optimizer
    require(isinstance(optimizer, pipeline.keras.optimizers.Adam) and
            int(optimizer.iterations.numpy()) == 1840 and
            math.isclose(float(optimizer.learning_rate.numpy()), 1e-4, rel_tol=1e-7),
            "Adam/1840pasos/tasaLR no se restauraron del checkpoint16.")
    require(len(optimizer.variables) == 98 and
            all(bool(np.all(np.isfinite(variable.numpy()))) for variable in model.weights) and
            all(bool(np.all(np.isfinite(variable.numpy()))) for variable in optimizer.variables),
            "Pesos/slotsAdam incompletos o no finitos.")
    with zipfile.ZipFile(RUN / "model_best.keras") as archive:
        with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as handle:
            saved = handle["optimizer/vars"]
            require(len(saved) == len(optimizer.variables) == 98,
                    "Checkpoint sin los98tensoresAdam guardados.")
            for index, variable in enumerate(optimizer.variables):
                require(np.array_equal(variable.numpy(), saved[str(index)][()]),
                        f"SlotAdam{index} no se restauro byte a byte en su valor numerico.")
    require(sha256(RUN / "model_best.keras") == CHECKPOINT_SHA,
            "Checkpoint fuente cambio durante su lectura.")
    return {
        "optimizer_class": type(optimizer).__name__, "optimizer_variables": 98,
        "optimizer_saved_loaded_values_exact": True, "optimizer_iterations": 1840,
        "optimizer_learning_rate": float(optimizer.learning_rate.numpy()),
        "model_parameters": model.count_params(), "trainable_parameters": 16588945,
        "trainable_variables": 48, "first_conv_kernel_shape": [3, 3, 4, 64],
        "encoder_trainable_convolutions": VGG_CONVS, "output_activation": "sigmoid",
        "all_model_and_optimizer_values_finite": True,
    }


def cumulative_history(prior_rows):
    rows = read_csv(RUN / "history.csv")
    require(rows[:INITIAL_EPOCH] == prior_rows and
            [int(row["epoch"]) for row in rows] == list(range(len(rows))) and
            INITIAL_EPOCH <= len(rows) <= TOTAL_EPOCHS,
            "CSV no conserva exactamente el prefijo16 o continuidad de epocas.")
    history = {key: [float(row[key]) for row in rows]
               for key in rows[0] if key != "epoch"}
    require(all(math.isfinite(value) for values in history.values() for value in values),
            "Historia acumulativa conNaN/Inf.")
    return rows, history


def prior_logged_duration():
    text = (ARCHIVE / "console_before_reboot.log").read_text(encoding="utf-8-sig")
    durations = [float(value) for value in re.findall(r"115/115\s+-\s+([0-9.]+)s\s+-", text)]
    require(len(durations) == INITIAL_EPOCH, "No se pudieron identificar las16duraciones previas.")
    return durations


def run():
    start = time.perf_counter()
    config, report, prior_rows, interruption = frozen_guards()
    pipeline = load_pipeline()
    tf, keras, np = pipeline.tf, pipeline.keras, pipeline.np
    tf.config.threading.set_intra_op_parallelism_threads(config["threads"])
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    args = pipeline.parser().parse_args([])
    for key, value in config.items():
        setattr(args, key, workspace_path(value) if key in ("dataset", "manifest", "output") else value)
    splits = confirm_splits(pipeline, args, read_csv(RUN / "splits.csv"), report)
    model = keras.models.load_model(RUN / "model_best.keras")
    checkpoint_evidence = confirm_loaded_checkpoint(pipeline, model)
    gradient = pipeline.gradient_check(model, splits["train"][0])
    require(gradient["variables_with_gradient"] == 48 and gradient["variables_without_gradient"] == 0,
            "No se verifican48gradientesencoder+decoder en el checkpoint restaurado.")
    fixed_x, fixed_y, _, _ = pipeline.read_pair(splits["train"][0])
    before_loss = float(model.loss(fixed_y[None], model(fixed_x[None], training=False)).numpy())
    decoder_before = [value.copy() for value in model.get_layer("water_probability").get_weights()]
    best_val = float(prior_rows[-1]["val_loss"])
    prior_durations = prior_logged_duration()
    context = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "interruption_reason": "operating_system_reboot",
        "interruption_record": str(ARCHIVE / "interrupcion.json"),
        "interruption_record_sha256": sha256(ARCHIVE / "interrupcion.json"),
        "system_boot_utc": interruption["system_boot_utc"], "archive_path": str(ARCHIVE),
        "initial_epoch": INITIAL_EPOCH, "next_epoch_one_based": 17,
        "total_epochs_requested": TOTAL_EPOCHS, "additional_epochs_maximum": 34,
        "partial_epoch17_work_discarded": True,
        "prior_history_sha256": HISTORY_SHA, "prior_checkpoint_sha256": CHECKPOINT_SHA,
        "prior_checkpoint_optimizer_iterations": 1840,
        "prior_checkpoint_validation_loss": best_val,
        "prior_checkpoint_best_epoch_one_based": 16,
        "pipeline_sha256": SOURCE_SHA, "helper_path": str(Path(__file__).resolve()),
        "helper_sha256": sha256(__file__),
        "prior_split_sha256": SPLITS_SHA, "dataset_manifest_sha256": MANIFEST_SHA,
        "dataset_summary_sha256": SUMMARY_SHA,
        "seed_original": 42, "seed_training_start": 58,
        "training_seed_rule": "original.make_dataset(train,4,True,42+16); generator increments once per iterator",
        "rng_continuity_limit": "Framework prefetch/process state was not serialized; bitwise uninterrupted equivalence is not claimed",
        "checkpoint_loaded_with_compile": True, "checkpoint_restoration": checkpoint_evidence,
        "callback_initial_state": {
            "ModelCheckpoint": {"monitor": "val_loss", "best": best_val, "save_best_only": True},
            "EarlyStopping": {"monitor": "val_loss", "best": best_val, "best_epoch": 15,
                              "wait": 0, "patience": 8, "restore_best_weights": True,
                              "best_weights_source": "loaded_checkpoint_epoch16"},
            "ReduceLROnPlateau": {"monitor": "val_loss", "best": best_val, "wait": 0,
                                  "cooldown_counter": 0, "factor": 0.5, "patience": 3,
                                  "min_delta": 1e-4, "min_lr": 1e-6},
        },
        "measurements_lost_on_restart": True,
        "initial_runtime_measurements_lost": ["initial_gradient_check", "initial_fixed_train_example_loss_before",
                                              "initial_transfer_equivalence_measurement", "initial_fit_start_monotonic_clock"],
        "test_evaluations_before_resume": 0, "known_previous_smoke_test_exposure": 4,
        "selection_rule": "minimum cumulative val_loss; threshold0.5 fixed; Test used only after selection",
        "prior_epoch_logged_seconds": prior_durations,
        "prior_epoch_logged_seconds_sum": sum(prior_durations),
    }
    class ResumedEarlyStopping(keras.callbacks.EarlyStopping):
        def on_train_begin(self, logs=None):
            super().on_train_begin(logs)
            self.best = best_val
            self.best_epoch = INITIAL_EPOCH - 1
            self.wait = 0
            self.best_weights = self.model.get_weights()

    class ResumedReduceLR(keras.callbacks.ReduceLROnPlateau):
        def on_train_begin(self, logs=None):
            super().on_train_begin(logs)
            self.best = best_val
            self.wait = 0
            self.cooldown_counter = 0

    checkpoint = keras.callbacks.ModelCheckpoint(
        str(RUN / "model_best.keras"), monitor="val_loss", save_best_only=True,
        initial_value_threshold=best_val)
    early = ResumedEarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    reduction = ResumedReduceLR(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

    def update_timing():
        observed = time.perf_counter() - start
        report["elapsed_resume_seconds"] = observed
        report["elapsed_seconds"] = observed + sum(prior_durations)
        report["elapsed_seconds_scope"] = "observed_resume_walltime_plus_prior_rounded_epoch_durations_excluding_offline_interval"
        report["training_segments"] = [
            {"segment": "before_reboot", "completed_epochs": 16,
             "epoch_duration_seconds_logged_rounded": prior_durations,
             "epoch_duration_seconds_sum": sum(prior_durations),
             "setup_seconds": None, "partial_epoch17_seconds": None},
            {"segment": "resumed_20261005", "walltime_seconds": observed,
             "includes_data_preflight_checkpoint_load_training_and_final_artifacts": True},
        ]
        report["offline_interval_in_elapsed_seconds"] = False

    report.update({
        "resumption": context, "resume_helper_sha256": context["helper_sha256"],
        "training_status": "started", "training_stage": "resume_pre_fit_checks_completed",
        "epochs_completed": INITIAL_EPOCH, "test_evaluations": 0,
        "input_shape": list(model.input_shape), "output_shape": list(model.output_shape),
        "model_parameters": model.count_params(), "trainable_parameters": 16588945,
        "gradient_check": gradient,
        "gradient_check_scope": "resume_loaded_checkpoint_pre_fit_train_batch",
        "initial_gradient_check": None, "initial_fixed_train_example_loss_before": None,
        "fixed_train_example_loss_before": before_loss,
        "fixed_train_example_loss_scope": "resume_loaded_checkpoint_vs_post_fit_same_train_example",
        "fixed_train_example_name": splits["train"][0]["name"],
        "weights_updated_scope": "resume_segment_loaded_checkpoint_vs_post_fit_weights",
        "weights_updated_scope_detail": "Comparison occurs after EarlyStopping restore_best_weights; unchanged selected best is allowed",
        "pretrained_transfer": {
            "source": "Keras VGG16 ImageNet include_top=False",
            "source_channels": "BGR after RGBpreprocess_input", "input_order": "B2,B3,B4,B8",
            "kernel_rgb_mapping": "B2blue->BGR0,B3green->BGR1,B4red->BGR2 direct",
            "nir_initialization": "mean_of_three_pretrained_kernels_unscaled_heuristic",
            "copied_conv_layers": VGG_CONVS, "encoder_frozen": False,
            "normalization": "DN/10000 before model",
            "model_preprocess": "normalized*255 minus BGRandNIRmeans",
            "means": pipeline.MEANS_BGR_NIR, "reflectance_clipping": False,
            "optical_kernels_unchanged": None,
            "first_conv_nir_zero_equivalence_max_abs_error": None,
            "measurement_status": "initial_runtime_values_lost_on_reboot",
            "evidence_scope": "source_algorithm_and_loaded_trained_checkpoint_not_initial_runtime_measurement",
            "domain_warning": "ImageNetnaturalcolors differ from satellite surface reflectance; NIRnotpretrained",
        },
    })
    report["history"] = cumulative_history(prior_rows)[1]
    report["limitations"] += [
        "Training interrupted during epoch17 by operating-system reboot; resumed from complete checkpoint16 includingAdam",
        "Initial runtime gradient/transfer/fixed-example diagnostics were lost; corresponding measurements here are scoped to resume",
        "Four of59 finalTest patches had previous smoke-test exposure; Test is not a wholly unseen independent benchmark",
        "FinalTest includes only1Imbabura and3Napo patches; regional metrics have limited sample support",
    ]
    atomic_json(RUN / "resume_context.json", context)
    update_timing()
    atomic_json(RUN / "report.json", report)

    class PersistResumeState(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            rows, history = cumulative_history(prior_rows)
            report["history"] = history
            report["epochs_completed"] = len(rows)
            report["training_stage"] = "resumed_fit"
            if epoch == INITIAL_EPOCH:
                current_decoder = self.model.get_layer("water_probability").get_weights()
                changed = any(not np.array_equal(before, after)
                              for before, after in zip(decoder_before, current_decoder))
                report["resume_training_weights_changed_during_fit"] = changed
                report["resume_training_updates_evidence"] = {
                    "scope": "first_resumed_epoch_end_before_any_on_train_end_best_weight_restoration",
                    "epoch_one_based": epoch + 1,
                    "decoder_output_weights_changed": changed,
                    "optimizer_iterations_before": 1840,
                    "optimizer_iterations_after": int(self.model.optimizer.iterations.numpy()),
                    "expected_optimizer_iterations_after": 1955,
                    "additional_model_forwards_used_for_this_comparison": 0,
                }
            report["callback_state_last_completed_epoch"] = {
                "epoch_zero_based": epoch, "checkpoint_best": float(checkpoint.best),
                "early_best": float(early.best), "early_best_epoch": int(early.best_epoch),
                "early_wait": int(early.wait), "early_stopped_epoch": int(early.stopped_epoch),
                "reduce_lr_best": float(reduction.best), "reduce_lr_wait": int(reduction.wait),
                "reduce_lr_cooldown_counter": int(reduction.cooldown_counter),
                "optimizer_iterations": int(model.optimizer.iterations.numpy()),
                "optimizer_learning_rate": float(model.optimizer.learning_rate.numpy()),
            }
            update_timing()
            atomic_json(RUN / "report.json", report)
            atomic_json(RUN / "resume_callback_state.json", report["callback_state_last_completed_epoch"])

    callbacks = [keras.callbacks.TerminateOnNaN(), checkpoint, early, reduction,
                 keras.callbacks.CSVLogger(str(RUN / "history.csv"), append=True), PersistResumeState()]
    train_data = pipeline.make_dataset(splits["train"], 4, True, 42 + INITIAL_EPOCH)
    val_data = pipeline.make_dataset(splits["val"], 4, False, 42)
    test_data = pipeline.make_dataset(splits["test"], 4, False, 42)
    try:
        print(json.dumps({"action": "resume_saved_checkpoint", "initial_epoch": 16,
                          "epochs_total_maximum": 50, "optimizer_iterations_restored": 1840,
                          "seed_training_start": 58, "test_evaluations": 0,
                          "checkpoint_sha256": CHECKPOINT_SHA}, ensure_ascii=False), flush=True)
        fit_start = time.perf_counter()
        history = model.fit(train_data, validation_data=val_data, initial_epoch=INITIAL_EPOCH,
                            epochs=TOTAL_EPOCHS, callbacks=callbacks, verbose=2, shuffle=False)
        report["resume_fit_seconds"] = time.perf_counter() - fit_start
        rows, history_values = cumulative_history(prior_rows)
        require(len(rows) == INITIAL_EPOCH + len(history.epoch), "Fit/CSV no coinciden en epocas reanudadas.")
        report["history"] = history_values
        report["epochs_completed"] = len(rows)
        report["epochs_completed_resume_segment"] = len(history.epoch)
        report["best_validation_epoch_one_based"] = int(np.argmin(history_values["val_loss"])) + 1
        decoder_after = model.get_layer("water_probability").get_weights()
        report["weights_updated"] = any(not np.array_equal(before, after)
                                        for before, after in zip(decoder_before, decoder_after))
        require(report.get("resume_training_weights_changed_during_fit") is True and
                report["resume_training_updates_evidence"]["optimizer_iterations_after"] == 1955,
                "No se verifico la actualizacion real de pesos/pasosAdam durante la primera epoca reanudada.")
        after_loss = float(model.loss(fixed_y[None], model(fixed_x[None], training=False)).numpy())
        require(math.isfinite(after_loss), "Perdida fija posterior no finita.")
        report["fixed_train_example_loss_after"] = after_loss
        report["fixed_train_example_loss_decreased"] = after_loss < before_loss
        best = keras.models.load_model(RUN / "model_best.keras")
        require(int(best.optimizer.iterations.numpy()) == 115 * report["best_validation_epoch_one_based"],
                "Adam del checkpoint elegido no corresponde al mejorVal acumulativo.")
        report["selected_checkpoint_sha256"] = sha256(RUN / "model_best.keras")
        report["selected_checkpoint_optimizer_iterations"] = int(best.optimizer.iterations.numpy())
        report["training_stage"] = "selected_best_validation_checkpoint_before_single_test"
        update_timing()
        atomic_json(RUN / "report.json", report)
        best.save(RUN / "model_final.keras")
        report["training_stage"] = "single_test_inference_started_do_not_repeat"
        update_timing()
        atomic_json(RUN / "report.json", report)
        test_result, cache = pipeline.evaluate_test_once(best, test_data, splits["test"])
        require(test_result["test_evaluations"] == 1 and test_result["test_inference_samples"] == 59,
                "Test no ejecuto exactamente un pase59.")
        np.savez_compressed(RUN / "test_probability_cache.npz", **cache)
        report.update(test_result)
        report["training_stage"] = "single_test_completed_exporting_cached_probabilities"
        report["test_probability_cache_sha256"] = sha256(RUN / "test_probability_cache.npz")
        update_timing()
        atomic_json(RUN / "report.json", report)
        pipeline.export_predictions(cache, splits["test"], RUN)
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for key in ("loss", "val_loss"):
            axes[0].plot(np.arange(1, len(history_values[key]) + 1), history_values[key], marker="o", label=key)
        for key in ("iou_water", "val_iou_water", "f1_water", "val_f1_water"):
            axes[1].plot(np.arange(1, len(history_values[key]) + 1), history_values[key], marker="o", label=key)
        for axis, title in zip(axes, ["BCE+Dice", "Metricas contra pseudomascara"]):
            axis.set(title=title, xlabel="Epoca total")
            axis.legend(fontsize=8)
            axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(RUN / "history.png", dpi=140)
        plt.close(figure)
        report["training_status"] = "completed"
        report["training_stage"] = "final_artifacts_completed"
        report["completion_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        update_timing()
        atomic_json(RUN / "report.json", report)
        print(json.dumps({key: report[key] for key in ("training_status", "epochs_completed",
                          "epochs_completed_resume_segment", "best_validation_epoch_one_based",
                          "test_metrics", "test_macro_regions", "weights_updated", "elapsed_seconds")},
                         ensure_ascii=False, indent=2), flush=True)
        return report
    except BaseException as error:
        report["training_status"] = "failed"
        report["error"] = repr(error)
        update_timing()
        atomic_json(RUN / "report.json", report)
        raise


if __name__ == "__main__":
    # O_EXCL impide dos continuaciones concurrentes sobre el mismo checkpoint.
    require(ARCHIVE.is_dir() and (ARCHIVE / "interrupcion.json").is_file(),
            "Falta archivo historico de interrupcion; no se escribe ningun archivo.")
    lock = RUN / ".resume_20261005.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, json.dumps({"pid": os.getpid(), "timestamp_utc":
                                        datetime.now(timezone.utc).isoformat()}).encode("utf-8"))
        os.close(descriptor)
        run()
    finally:
        lock.unlink(missing_ok=True)
