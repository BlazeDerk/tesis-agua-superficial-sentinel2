"""U-Net + VGG16 ImageNet, cuatro canales Sentinel-2, entrenamiento reproducible.

Inputs del modelo = [B2,B3,B4,B8]/10000. Puente explicito a VGG: primeros
tres canales ya BGR, x255 y medias ImageNet BGR; NIR usa media de esas medias.
Kernel ImageNet BGR se copia directamente; NIR se inicializa con la media de
los tres kernels, heuristica sin preentrenamiento NIR. No clipping radiometrico.
La prueba smoke usa legado existente. smoke-multiregional requiere la salida
real de fase2 y muestrea despues de dividir todo el dataset espacialmente.
Ninguna prueba corta acredita generalizacion. Las mascaras son pseudoetiquetas.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / "entrenamiento" / "python_deps"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(ROOT / "entrenamiento" / "keras_cache"))
if os.name == "nt":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # TF moderno nativo Windows: CPU.

import numpy as np
import rasterio
import tensorflow as tf
from tensorflow import keras

PATCH = 256
NODATA = 65535
CRS = rasterio.crs.CRS.from_epsg(32717)
SUSPECTED_CLOUDS = {"parche_3.tif", "parche_213.tif", "parche_491.tif", "parche_290.tif"}
MULTIREGIONAL_ZONES = {"el_oro", "golfo_guayaquil", "lagos_imbabura", "rio_napo"}
SMOKE_MODES = {"smoke", "smoke-multiregional"}
MEANS_BGR_NIR = [103.939, 116.779, 123.68, (103.939 + 116.779 + 123.68) / 3]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_pair(record):
    with rasterio.open(record["image_path"]) as image, rasterio.open(record["mask_path"]) as mask:
        x, y = image.read(), mask.read(1)
        require(x.shape == (4, PATCH, PATCH) and y.shape == (PATCH, PATCH), f"Dimensiones incorrectas: {record['name']}")
        require(set(image.dtypes + mask.dtypes) == {"uint16"}, f"uint16 requerido: {record['name']}")
        require(image.crs == mask.crs == CRS and image.transform.almost_equals(mask.transform, precision=1e-7),
                f"Pareja desalineada/CRS incorrecto: {record['name']}")
        require(np.allclose([image.transform.a, image.transform.b, image.transform.d, image.transform.e],
                            [10, 0, 0, -10], rtol=0, atol=1e-7), f"Resolucion/grilla incorrecta: {record['name']}")
        require(not np.any(image.read_masks() == 0) and not np.any(mask.read_masks() == 0)
                and not np.any(x == NODATA) and not np.any(y == NODATA), f"NoData en {record['name']}")
        require(bool(np.all(np.isin(y, [0, 1]))), f"Mascara no binaria: {record['name']}")
        return np.moveaxis(x, 0, -1).astype("float32") / 10000.0, y[..., None].astype("float32"), image.transform, list(image.bounds)


def resolve_path(value, dataset, folder, basename):
    candidates = [dataset / folder / basename]
    if value:
        candidates += [dataset / value, Path(value)]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise ValueError(f"No existe {folder}/{basename} en {dataset}")


def suspected_cloud_legacy(source, name):
    """Identifica nombres originales incluso despues de renombrar en fase2.

    Paths Windows del manifiesto tambien se interpretan correctamente en Colab.
    Una fuente nueva no hereda alertas solo por una coincidencia de nombre.
    """
    if source.get("kind") == "new":
        return False
    if source.get("kind") not in (None, "", "legacy") and source.get("zone") != "el_oro":
        return False
    original = str(source.get("source_image") or "").replace("\\", "/").rsplit("/", 1)[-1].casefold()
    current = str(name).replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return current in SUSPECTED_CLOUDS or original in SUSPECTED_CLOUDS


def load_records(args):
    dataset = args.dataset.resolve()
    require(dataset.is_dir(), f"Dataset ausente: {dataset}; obtener y auditar mosaicos GEE antes de entrenamiento final.")
    manifest = args.manifest
    if manifest is None and (dataset / "manifest.csv").is_file():
        manifest = dataset / "manifest.csv"
    multiregional = args.mode in ("final", "smoke-multiregional")
    if multiregional:
        # Impide presentar un entrenamiento solo con El Oro como final del
        # proyecto multirregional. El manifiesto publica la procedencia por par.
        require(manifest is not None and manifest.is_file(),
                "Final/smoke-multiregional requieren manifest.csv generado por fase2 real.")
        summary_path = dataset / "summary.json"
        require(summary_path.is_file(), "Final/smoke-multiregional requieren summary.json de fase2 real.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(summary.get("phase") == 2 and summary.get("training_executed") is False,
                "summary.json no acredita el recorte/unificacion de fase2.")
        require(summary.get("accepted_new", 0) > 0 and summary.get("accepted_legacy", 0) > 0,
                "Final requiere parches nuevos y legado El Oro unificados.")
        require(all(summary.get("accepted_by_zone", {}).get(zone, 0) > 0 for zone in
                    MULTIREGIONAL_ZONES),
                "Final requiere parches utiles de las tres zonas nuevas y El Oro.")
    require(args.mode not in SMOKE_MODES or manifest is not None,
            "Smoke requiere --manifest explicito (preferir manifiesto_384_sin_cuatro_ceros.csv).")
    if manifest:
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
    else:
        source_rows = [{"name": path.name} for path in sorted((dataset / "images").glob("*.tif"))]
    require(bool(source_rows), "Dataset/manifiesto sin pares.")
    if multiregional:
        require(len(source_rows) == summary.get("accepted_total"),
                "Conteo de manifiesto no coincide con summary.json de fase2.")
        manifest_counts = Counter(row.get("zone") for row in source_rows)
        require(dict(manifest_counts) == summary.get("accepted_by_zone"),
                "Procedencia por zona/manifiesto difiere de summary.json.")
        require(all(row.get("output_image_sha256") and row.get("output_mask_sha256") for row in source_rows),
                "Final requiere huellas de los TIFF publicados por fase2.")
    # El manifiesto legacy conserva el hash de la mascara ORIGINALuint8 como
    # procedencia. La copia corregidauint16 tiene otro hash; no confundirlos.
    corrected_mask_hashes = {}
    correction_csv = manifest.parent / "correcciones_verificadas.csv" if manifest else None
    if correction_csv and correction_csv.is_file():
        with correction_csv.open(encoding="utf-8-sig", newline="") as handle:
            corrected_mask_hashes = {row["name"]: row["corrected_mask_sha256"] for row in csv.DictReader(handle)}
    records, issues, hashes, excluded_cloud_sources = [], Counter(), set(), []
    for source in source_rows:
        name = source.get("name") or Path(source.get("image", "")).name
        require(bool(name), "Manifiesto sin name/image.")
        if suspected_cloud_legacy(source, name):
            issues["visually_suspected_cloud_example_excluded"] += 1
            excluded_cloud_sources.append({"published_name": name, "source_image": source.get("source_image", name),
                                           "reason": "visually_suspected_cloud_in_legacy_audit"})
            continue
        image = resolve_path(source.get("image") or source.get("image_path"), dataset, "images", name)
        mask = resolve_path(source.get("mask") or source.get("mask_path"), dataset, "masks", name)
        record = {"name": name, "image_path": str(image), "mask_path": str(mask),
                  "zone": source.get("zone", "el_oro"), "label_type": source.get("label_type", "MNDWI_pseudolabel"),
                  "source_image": source.get("source_image", str(image))}
        x, y, transform, bounds = read_pair(record)
        zeros = int(np.count_nonzero(np.all(x == 0, axis=-1)))
        if zeros and not args.allow_ambiguous_zeros:
            issues["all_four_zero_pixels_excluded"] += 1
            continue
        require(int(y.sum()) >= 50, f"Parche <50 pixeles de agua: {name}")
        image_hash, mask_hash = sha256(image), sha256(mask)
        expected_image = source.get("image_source_sha256") or source.get("output_image_sha256")
        expected_mask = source.get("output_mask_sha256") or source.get("corrected_mask_sha256") or corrected_mask_hashes.get(name)
        if source.get("mask_source_sha256") and not expected_mask:
            issues["legacy_original_mask_hash_provenance_only_current_hash_recorded"] += 1
        require(not expected_image or expected_image == image_hash, f"SHA256 imagen/manifiesto difiere: {name}")
        require(not expected_mask or expected_mask == mask_hash, f"SHA256 mascara/manifiesto difiere: {name}")
        digest = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
        require(digest not in hashes, f"Pareja duplicada exacta: {name}")
        hashes.add(digest)
        descriptions = None
        with rasterio.open(image) as src:
            descriptions = list(src.descriptions)
        if any(descriptions):
            require(descriptions == ["B2", "B3", "B4", "B8"], f"Orden de bandas diferente: {name}")
        else:
            issues["band_order_from_user_context_not_embedded"] += 1
        record.update({"bounds": bounds, "transform": list(transform)[:6], "source_sha256": image_hash,
                       "mask_sha256": mask_hash, "water_pixels": int(y.sum()), "all_zero_pixels": zeros,
                       "spatial_group": f"EPSG32717_10240m_{math.floor((bounds[0]+bounds[2])/20480)}_"
                                        f"{math.floor((bounds[1]+bounds[3])/20480)}"})
        records.append(record)
    require(len(records) >= 10, "Se requieren al menos10 pares admisibles para la prueba/split.")
    if multiregional:
        require({row["zone"] for row in records} == MULTIREGIONAL_ZONES,
                "El filtrado deja alguna zona sin pares admisibles; auditar/ampliar antes de final.")
    for i, left in enumerate(records):
        for right in records[:i]:
            a, b = left["bounds"], right["bounds"]
            require(not (min(a[2], b[2]) - max(a[0], b[0]) > 1e-7
                         and min(a[3], b[3]) - max(a[1], b[1]) > 1e-7),
                    f"Footprints superpuestos: {left['name']}, {right['name']}")
    issue_report = dict(issues)
    issue_report["known_cloud_excluded_sources"] = excluded_cloud_sources
    return records, issue_report, str(manifest.resolve()) if manifest else None


def geographic_distance(a, b):
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


def sample_by_zone(records, sample_count, rng):
    """Seleccion por procedencia, con al menos un ejemplo de cada zona."""
    by_zone = {zone: [row for row in records if row["zone"] == zone]
               for zone in sorted({row["zone"] for row in records})}
    require(len(by_zone) <= sample_count <= len(records), "Muestra insuficiente/excesiva para representar cada zona.")
    remaining_capacity = {zone: len(rows)-1 for zone, rows in by_zone.items()}
    extra = sample_count - len(by_zone)
    total_capacity = sum(remaining_capacity.values())
    quotas = {zone: extra*capacity/total_capacity if total_capacity else 0.0
              for zone, capacity in remaining_capacity.items()}
    allocated = {zone: 1+int(math.floor(quota)) for zone, quota in quotas.items()}
    residual = sample_count - sum(allocated.values())
    tie_break = {zone: float(rng.random()) for zone in by_zone}
    priority = sorted(by_zone, key=lambda zone: (-(quotas[zone]-math.floor(quotas[zone])), tie_break[zone]))
    for zone in priority:
        if residual and allocated[zone] < len(by_zone[zone]):
            allocated[zone] += 1
            residual -= 1
    require(residual == 0, "No se pudo completar muestra por zona.")
    selected = []
    for zone, rows in by_zone.items():
        chosen = rng.choice(len(rows), size=allocated[zone], replace=False)
        selected.extend(rows[index] for index in sorted(chosen))
    return selected


def split_geographic(records, seed, buffer_m, smoke_samples=None, stratify_by_zone=False, return_population=False):
    groups = defaultdict(list)
    for record in records:
        groups[record["spatial_group"]].append(record)
    require(len(groups) >= 3, "Menos de3 grupos geograficos: proporcionar mas regiones, no dividir vecinos aleatoriamente.")
    # Sectores ordenados geograficamente mantienen menos fronteras que grupos
    # dispersos al azar. Semilla desempata grupos con coordenada X identica.
    rng = np.random.default_rng(seed)
    tie_break = {key: float(rng.random()) for key in sorted(groups)}
    ordered = sorted(groups, key=lambda key: (min(r["bounds"][0] for r in groups[key]), tie_break[key]))
    result = {"train": [], "val": [], "test": []}
    if stratify_by_zone:
        # Final: todas las regiones aportan Train/Val/Test con grupos enteros.
        # Dos cortes de sectores minimizan desviacion80/10/10 por zona ANTES
        # de buffer. No se promete ratio exacto incompatible con bloques.
        for zone in sorted({record["zone"] for record in records}):
            zone_keys = [key for key in ordered if any(row["zone"] == zone for row in groups[key])]
            require(len(zone_keys) >= 3, f"Zona {zone}: menos de3 bloques de10.24km; ampliar datos antes de final.")
            require(all(all(row["zone"] == zone for row in groups[key]) for key in zone_keys),
                    "Un bloque espacial contiene dos zonas: conservarlo unido requiere redefinir particion.")
            cumulative = np.cumsum([0] + [len(groups[key]) for key in zone_keys])
            total = int(cumulative[-1])
            candidates = []
            for left in range(1, len(zone_keys)-1):
                for right in range(left+1, len(zone_keys)):
                    test_count, val_count = int(cumulative[left]), int(cumulative[right]-cumulative[left])
                    train_count = total - int(cumulative[right])
                    deviation = abs(test_count/total-0.1) + abs(val_count/total-0.1) + abs(train_count/total-0.8)
                    candidates.append((deviation, -train_count, left, right))
            _, _, left, right = min(candidates)
            for index, key in enumerate(zone_keys):
                split = "test" if index < left else "val" if index < right else "train"
                result[split].extend(dict(row, split=split) for row in groups[key])
    else:
        target = max(1, round(len(records) * 0.10))
        for key in ordered:
            split = "test" if len(result["test"]) < target else "val" if len(result["val"]) < target else "train"
            for row in groups[key]:
                result[split].append(dict(row, split=split))
    discarded = []
    for split, protected in (("val", ["test"]), ("train", ["test", "val"])):
        protect = [record for other in protected for record in result[other]]
        retained = []
        for record in result[split]:
            if any(geographic_distance(record["bounds"], other["bounds"]) <= buffer_m + 1e-7 for other in protect):
                discarded.append(dict(record, reason="geographic_boundary_buffer"))
            else:
                retained.append(record)
        result[split] = retained
    require(all(result.values()), "Buffer deja algun split vacio: ajustar regiones/buffer documentadamente.")
    if stratify_by_zone:
        expected_zones = {record["zone"] for record in records}
        require(all({record["zone"] for record in rows} == expected_zones for rows in result.values()),
                "Buffer deja alguna zona sin Train/Val/Test: ampliar region o revisar buffer documentadamente.")
    population = {split: list(rows) for split, rows in result.items()}
    if smoke_samples is not None:
        if stratify_by_zone:
            zones_count = len({row["zone"] for row in records})
            require(3*zones_count <= smoke_samples <= 80, "Smoke multirregional requiere12 a80 parches para las cuatro zonas.")
        else:
            zones_count = 1
            require(8 <= smoke_samples <= 16, "Smoke legado debe usar8 a16 parches.")
        wanted = {"val": max(zones_count, round(smoke_samples * 0.10)), "test": max(zones_count, round(smoke_samples * 0.10))}
        wanted["train"] = smoke_samples - wanted["val"] - wanted["test"]
        for split in result:
            require(len(result[split]) >= wanted[split], f"No hay suficientes parches {split} trasbuffer.")
            if stratify_by_zone:
                result[split] = sample_by_zone(result[split], wanted[split], rng)
            else:
                chosen = rng.choice(len(result[split]), size=wanted[split], replace=False)
                result[split] = [result[split][i] for i in sorted(chosen)]
    for i, a_name in enumerate(("train", "val", "test")):
        for b_name in ("train", "val", "test")[:i]:
            require(not ({r["spatial_group"] for r in result[a_name]} & {r["spatial_group"] for r in result[b_name]}),
                    "Grupo geografico compartido entre splits.")
            require(all(geographic_distance(a["bounds"], b["bounds"]) > buffer_m
                        for a in result[a_name] for b in result[b_name]), "Fronteras demasiado cercanas entre splits.")
    return (result, discarded, population) if return_population else (result, discarded)


@keras.utils.register_keras_serializable(package="Tesis")
class VGG4Preprocess(keras.layers.Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32) * 255.0 - tf.constant(MEANS_BGR_NIR, dtype=tf.float32)


@keras.utils.register_keras_serializable(package="Tesis")
class BCEDice(keras.losses.Loss):
    def __init__(self, name="bce_dice", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        truth, prediction = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
        bce = tf.reduce_mean(keras.losses.binary_crossentropy(truth, prediction), axis=(1, 2))
        intersection = tf.reduce_sum(truth * prediction, axis=(1, 2, 3))
        denominator = tf.reduce_sum(truth + prediction, axis=(1, 2, 3))
        dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
        return 0.5 * bce + 0.5 * (1.0 - dice)


@keras.utils.register_keras_serializable(package="Tesis")
class WaterConfusionMetric(keras.metrics.Metric):
    def __init__(self, mode="iou", threshold=0.5, name=None, **kwargs):
        super().__init__(name=name or ("iou_water" if mode == "iou" else "f1_water"), **kwargs)
        self.mode, self.threshold = mode, threshold
        self.tp = self.add_weight(name="tp", initializer="zeros", dtype="float64")
        self.fp = self.add_weight(name="fp", initializer="zeros", dtype="float64")
        self.fn = self.add_weight(name="fn", initializer="zeros", dtype="float64")

    def update_state(self, y_true, y_pred, sample_weight=None):
        truth = tf.cast(y_true >= 0.5, tf.float64)
        predicted = tf.cast(y_pred >= self.threshold, tf.float64)
        if sample_weight is None:
            weight = tf.ones_like(truth)
        else:
            weight = tf.cast(sample_weight, tf.float64)
            trailing_axes = tf.maximum(tf.rank(truth) - tf.rank(weight), 0)
            weight = tf.reshape(weight, tf.concat([tf.shape(weight), tf.ones([trailing_axes], tf.int32)], axis=0))
        self.tp.assign_add(tf.reduce_sum(weight * truth * predicted))
        self.fp.assign_add(tf.reduce_sum(weight * (1.0 - truth) * predicted))
        self.fn.assign_add(tf.reduce_sum(weight * truth * (1.0 - predicted)))

    def result(self):
        if self.mode == "iou":
            return tf.cast(tf.math.divide_no_nan(self.tp, self.tp + self.fp + self.fn), tf.float32)
        return tf.cast(tf.math.divide_no_nan(2 * self.tp, 2 * self.tp + self.fp + self.fn), tf.float32)

    def reset_state(self):
        for variable in (self.tp, self.fp, self.fn):
            variable.assign(0)

    def get_config(self):
        return dict(super().get_config(), mode=self.mode, threshold=self.threshold)


def build_model(freeze_encoder=False, learning_rate=1e-4):
    original = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(PATCH, PATCH, 3))
    encoder = keras.applications.VGG16(weights=None, include_top=False, input_shape=(PATCH, PATCH, 4))
    transferred = []
    for layer in encoder.layers:
        if not layer.weights:
            continue
        pretrained = original.get_layer(layer.name).get_weights()
        if layer.name == "block1_conv1":
            kernel, bias = pretrained
            kernel4 = np.concatenate([kernel, np.mean(kernel, axis=2, keepdims=True)], axis=2)
            layer.set_weights([kernel4, bias])
            require(np.array_equal(layer.get_weights()[0][:, :, :3], kernel), "Transferencia BGRfirstkernel difiere.")
        else:
            layer.set_weights(pretrained)
        layer.trainable = not freeze_encoder
        transferred.append(layer.name)
    # Prueba de equivalencia firstConv: NIR centrado0; RGBreal de ImageNet pasa
    # preprocess_input RGB->BGR; B2/B3/B4 recibe exactamente ese mismo BGR.
    rgb = np.random.default_rng(1).uniform(0, 255, (1, 16, 16, 3)).astype("float32")
    preprocessed = keras.applications.vgg16.preprocess_input(rgb.copy())
    fourth = np.zeros((1, 16, 16, 1), dtype="float32")
    expected = original.get_layer("block1_conv1")(preprocessed).numpy()
    actual = encoder.get_layer("block1_conv1")(np.concatenate([preprocessed, fourth], axis=-1)).numpy()
    difference = float(np.max(np.abs(expected - actual)))
    require(np.allclose(expected, actual, rtol=1e-5, atol=1e-4), "Equivalencia VGG3->4 con NIR0 fallo.")
    inputs = keras.Input((PATCH, PATCH, 4), name="sentinel_B2_B3_B4_B8_div10000")
    features = keras.Model(encoder.input, [encoder.get_layer(name).output for name in (
        "block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3", "block5_pool")], name="vgg16_encoder4")
    s1, s2, s3, s4, s5, bottom = features(VGG4Preprocess(name="vgg_bgr_nir_affine")(inputs))
    x = bottom
    for index, (skip, filters) in enumerate(zip([s5, s4, s3, s2, s1], [128, 64, 32, 16, 16]), 1):
        x = keras.layers.UpSampling2D(2, interpolation="bilinear", name=f"decoder{index}_upsample")(x)
        x = keras.layers.Concatenate(name=f"decoder{index}_skip")([x, skip])
        x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal", name=f"decoder{index}_conv1")(x)
        x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal", name=f"decoder{index}_conv2")(x)
    output = keras.layers.Conv2D(1, 1, activation="sigmoid", name="water_probability")(x)
    model = keras.Model(inputs, output, name="UNet_VGG16_Sentinel2_4band")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=BCEDice(),
                  metrics=[WaterConfusionMetric("iou"), WaterConfusionMetric("f1")])
    transfer = {"source": "Keras VGG16 ImageNet include_top=False", "source_channels": "BGR after RGBpreprocess_input",
                "input_order": "B2,B3,B4,B8", "kernel_rgb_mapping": "B2blue->BGR0,B3green->BGR1,B4red->BGR2 direct",
                "nir_initialization": "mean_of_three_pretrained_kernels_unscaled_heuristic",
                "optical_kernels_unchanged": True, "copied_conv_layers": transferred,
                "normalization": "DN/10000 before model", "model_preprocess": "normalized*255 minus BGRandNIRmeans",
                "means": MEANS_BGR_NIR, "reflectance_clipping": False,
                "first_conv_nir_zero_equivalence_max_abs_error": difference,
                "encoder_frozen": freeze_encoder, "domain_warning": "ImageNetnaturalcolors differ from satellite surface reflectance; NIRnotpretrained"}
    del original
    return model, transfer


def paired_augmentation(x, y, rng):
    """Mismos giros/flips discretos; preserva alineacion y etiquetas binarias."""
    rotations = int(rng.integers(0, 4))
    x, y = np.rot90(x, rotations, axes=(0, 1)), np.rot90(y, rotations, axes=(0, 1))
    for axis in (0, 1):
        if rng.random() < 0.5:
            x, y = np.flip(x, axis=axis), np.flip(y, axis=axis)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def make_dataset(records, batch_size, training, seed):
    """Para Val/Test, training=False conserva EXACTAMENTE el orden de records.

    No shuffle, augmentation ni drop_remainder en Test. evaluate_test_once usa
    ese orden para asociar cada fila de un lote a su zona y fuente en records.
    No agregar shuffle/repeat/filtros a test_data despues de esta funcion.
    """
    epoch = [0]
    def generate():
        rng = np.random.default_rng(seed + epoch[0])
        epoch[0] += 1
        indices = rng.permutation(len(records)) if training else np.arange(len(records))
        for index in indices:
            x, y, _, _ = read_pair(records[index])
            if training:
                x, y = paired_augmentation(x, y, rng)
            yield np.ascontiguousarray(x), np.ascontiguousarray(y)
    dataset = tf.data.Dataset.from_generator(generate, output_signature=(
        tf.TensorSpec((PATCH, PATCH, 4), tf.float32), tf.TensorSpec((PATCH, PATCH, 1), tf.float32)))
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(records)))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options).batch(batch_size).prefetch(1)


def gradient_check(model, record):
    x, y, _, _ = read_pair(record)
    with tf.GradientTape() as tape:
        prediction = model(x[None], training=True)
        loss = model.loss(y[None], prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    real = [gradient for gradient in gradients if gradient is not None]
    finite = bool(real) and all(bool(tf.reduce_all(tf.math.is_finite(g)).numpy()) for g in real)
    norm = float(tf.linalg.global_norm(real).numpy())
    require(finite and math.isfinite(float(loss.numpy())) and norm > 0, "Gradientes no finitos/vacios/ceros.")
    return {"finite": finite, "global_norm": norm, "loss": float(loss.numpy()),
            "variables_with_gradient": len(real), "variables_without_gradient": len(gradients)-len(real)}


def select_prediction_examples(records, limit=4):
    """Regla fija: una fila por zona primero; completa con el orden existente."""
    selected = [next(row for row in records if row.get("zone", "el_oro") == zone)
                for zone in sorted({row.get("zone", "el_oro") for row in records})][:limit]
    selected_names = {row["name"] for row in selected}
    return selected + [row for row in records if row["name"] not in selected_names][:max(0, limit-len(selected))]


def confusion_summary(counts):
    """Agrega ratios con la convencion de WaterConfusionMetric: cero si0/0."""
    result = dict(counts)
    iou_denominator = counts["tp"] + counts["fp"] + counts["fn"]
    f1_denominator = 2*counts["tp"] + counts["fp"] + counts["fn"]
    # WaterConfusionMetric acumula float64 y devuelve float32. Mantener ese
    # redondeo en los campos compatibles con Keras.evaluate.
    result["iou_water"] = float(np.float32(counts["tp"]/iou_denominator if iou_denominator else 0.0))
    result["f1_water"] = float(np.float32(2*counts["tp"]/f1_denominator if f1_denominator else 0.0))
    return result


def evaluate_test_once(model, test_data, records, prediction_limit=4):
    """Un solo pase Test; mismo Tensor para loss, conteos y cache de salidas.

    Contrato: test_data debe provenir de make_dataset(records, ..., False, ...)
    y recorrer exactamente records, sin shuffle/repeat/augmentation/filtros.
    Solo pares (x,y), todos los pixeles validos segun read_pair. La arquitectura
    de este proyecto usa la estrategia normal de una replica (CPU o una GPU).

    Keras3 test_step llama compute_loss con training=False y acumula su escalar
    en Mean ponderada por el tamano de lote. compute_loss incluye loss compilado
    y regularizadores. Se pasa y_pred ORIGINAL antes de .numpy para conservar
    _keras_logits y la BCE estable cuando Sigmoid satura.
    """
    require(bool(records) and len({row["name"] for row in records}) == len(records),
            "Test requiere filas no vacias y nombres unicos.")
    require(model.distribute_strategy.num_replicas_in_sync == 1,
            "Evaluacion Test de este proyecto requiere estrategia de una replica.")
    selected_names = {row["name"] for row in select_prediction_examples(records, prediction_limit)}
    probability_cache, by_zone = {}, {}
    fields = ("n_patches", "valid_pixels", "water_pixels", "predicted_water_pixels", "tp", "fp", "fn", "tn")
    global_counts = {key: 0 for key in fields}
    loss_tracker = keras.metrics.Mean(name="loss", dtype=keras.backend.floatx())
    offset, batch_sizes = 0, []
    for batch in test_data:
        require(isinstance(batch, (tuple, list)) and len(batch) == 2, "Test espera lotes (x,y), sin sample_weight.")
        batch_x, batch_y = batch
        batch_size_tensor = tf.shape(batch_x)[0]
        batch_size = int(batch_size_tensor.numpy())
        require(batch_size > 0 and offset+batch_size <= len(records), "Test contiene mas filas que records.")
        prediction_tensor = model(batch_x, training=False)  # Unico forward para estas filas.
        batch_loss = model.compute_loss(x=batch_x, y=batch_y, y_pred=prediction_tensor,
                                        sample_weight=None, training=False)
        require(int(tf.rank(batch_loss).numpy()) == 0 and math.isfinite(float(batch_loss.numpy())),
                "Perdida Test compilada no escalar/no finita.")
        loss_tracker.update_state(batch_loss, sample_weight=batch_size_tensor)
        # La conversion ocurre DESPUES de compute_loss: no reconstruir Sigmoid.
        probabilities = np.asarray(prediction_tensor.numpy())
        labels = np.asarray(batch_y.numpy())
        require(probabilities.shape == labels.shape and probabilities.ndim == 4 and probabilities.shape[-1] == 1,
                "Dimensiones Test incorrectas: requiere [batch,height,width,1].")
        require(bool(np.all(np.isfinite(probabilities))) and bool(np.all((probabilities >= 0) & (probabilities <= 1)))
                and bool(np.all(np.isin(labels, [0, 1]))), "Probabilidad/etiqueta Test invalida.")
        truth, predicted = labels >= 0.5, probabilities >= 0.5
        for index, record in enumerate(records[offset:offset+batch_size]):
            real, hard = truth[index], predicted[index]
            counted = {"n_patches": 1, "valid_pixels": int(real.size), "water_pixels": int(np.count_nonzero(real)),
                       "predicted_water_pixels": int(np.count_nonzero(hard)),
                       "tp": int(np.count_nonzero(real & hard)), "fp": int(np.count_nonzero(~real & hard)),
                       "fn": int(np.count_nonzero(real & ~hard)), "tn": int(np.count_nonzero(~real & ~hard))}
            zone = record.get("zone", "el_oro")
            regional = by_zone.setdefault(zone, {key: 0 for key in fields})
            for key in fields:
                regional[key] += counted[key]
                global_counts[key] += counted[key]
            if record["name"] in selected_names:
                probability_cache[record["name"]] = probabilities[index, ..., 0].astype("float32", copy=True)
        offset += batch_size
        batch_sizes.append(batch_size)
    require(offset == len(records) and set(probability_cache) == selected_names,
            "Test incompleto o cache de exportaciones incompleto.")
    summarized_global = confusion_summary(global_counts)
    summarized_zones = {zone: confusion_summary(counts) for zone, counts in sorted(by_zone.items())}
    result = {
        "test_metrics": {"loss": float(loss_tracker.result().numpy()), "iou_water": summarized_global["iou_water"],
                         "f1_water": summarized_global["f1_water"]},
        "test_confusion": summarized_global, "test_by_zone": summarized_zones,
        "test_macro_regions": {"iou_water": float(np.mean([row["iou_water"] for row in summarized_zones.values()])),
                               "f1_water": float(np.mean([row["f1_water"] for row in summarized_zones.values()])),
                               "n_zones": len(summarized_zones), "aggregation": "unweighted_mean_of_region_metrics",
                               "zero_denominator_value": 0.0},
        "test_evaluations": 1, "test_inference_batches": len(batch_sizes), "test_inference_samples": offset,
        "test_batch_sizes": batch_sizes,
        "test_loss_aggregation": "compiled_compute_loss_including_regularizers_then_Mean_weighted_by_batch_size",
        "test_order_contract": "make_dataset_training_false_exact_records_order_no_shuffle_repeat_or_filter",
        "test_export_probability_source": "cached_from_same_single_test_inference_pass",
    }
    return result, probability_cache


def export_predictions(probability_cache, records, output, limit=4):
    """Exporta cache del unico pase Test; no recibe ni ejecuta el modelo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    pred_dir = output / "predictions"
    pred_dir.mkdir()
    selected = select_prediction_examples(records, limit)
    figure, axes = plt.subplots(len(selected), 4, figsize=(14, 3.3 * len(selected)), squeeze=False)
    for index, record in enumerate(selected):
        x, y, transform, _ = read_pair(record)
        require(record["name"] in probability_cache, "Falta probabilidad del pase Test; no se vuelve a inferir.")
        probability = np.asarray(probability_cache[record["name"]], dtype="float32")
        require(probability.shape == y.shape[:2], "Cache de probabilidad desalineado con el parche.")
        mask = (probability >= 0.5).astype("uint16")
        require(bool(np.all(np.isfinite(probability))) and np.all((probability >= 0) & (probability <= 1)), "Probabilidad invalida.")
        for suffix, data, dtype in (("probability", probability, "float32"), ("mask", mask, "uint16")):
            with rasterio.open(pred_dir / f"{Path(record['name']).stem}_{suffix}.tif", "w", driver="GTiff",
                               height=probability.shape[0], width=probability.shape[1], count=1, dtype=dtype, crs=CRS, transform=transform,
                               compress="deflate") as target:
                target.write(data, 1)
                target.set_band_description(1, "water_probability" if suffix == "probability" else "water_mask_predicted")
                target.update_tags(source_image_sha256=record["source_sha256"], threshold="0.5", prediction_type="model_not_ground_truth")
        rgb = x[..., [2, 1, 0]]
        low, high = np.percentile(rgb, [2, 98])
        shown = np.clip((rgb - low) / max(float(high-low), 1e-6), 0, 1)  # Solo visualizacion.
        for axis, data, title, kwargs in zip(axes[index], [shown, y[..., 0], probability, mask],
                                           ["RGB (estiramiento visual)", "Pseudoetiqueta MNDWI", "Probabilidad agua", "Prediccion >=0.5"],
                                           [{}, {"cmap": "Blues", "vmin": 0, "vmax": 1},
                                            {"cmap": "viridis", "vmin": 0, "vmax": 1}, {"cmap": "Blues", "vmin": 0, "vmax": 1}]):
            axis.imshow(data, **kwargs)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        axes[index, 0].set_title(textwrap.fill(record["name"], width=38)+"\n"+record.get("zone", "el_oro")+" · RGB", fontsize=9)
    figure.suptitle("Predicciones U-Net: referencia espectral MNDWI, sin referencia independiente", fontsize=12)
    figure.tight_layout()
    figure.savefig(output / "predictions.png", dpi=140)
    plt.close(figure)


def run(args):
    start = time.perf_counter()
    require(not args.output.exists(), f"Salida existente: {args.output}; usar nueva carpeta, no sobrescribir.")
    require(args.epochs > 0 and args.batch_size > 0 and args.buffer_m >= 0, "Parametros invalidos.")
    require(args.threads > 0 and math.isfinite(args.learning_rate) and args.learning_rate > 0,
            "Hilos/tasa de aprendizaje invalidos.")
    require(args.mode not in SMOKE_MODES or 1 <= args.epochs <= 3, "Smoke requiere entre1 y3 epocas.")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        # Notebook con runtime ya inicializado: conservar configuracion actual.
        pass
    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    records, issues, manifest = load_records(args)
    smoke_samples = args.smoke_samples if args.smoke_samples is not None else (40 if args.mode == "smoke-multiregional" else 12)
    multiregional = args.mode in ("final", "smoke-multiregional")
    splits, discarded, population = split_geographic(records, args.seed, args.buffer_m,
                                         smoke_samples if args.mode in SMOKE_MODES else None,
                                         stratify_by_zone=multiregional, return_population=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    used = [row for split in ("train", "val", "test") for row in splits[split]]
    for filename, rows in (("splits.csv", used), ("spatial_partition.csv", [row for split in population for row in population[split]])):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "image_path", "mask_path", "source_image", "zone", "label_type", "split", "spatial_group", "bounds", "source_sha256", "mask_sha256"])
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(row[key]) if key == "bounds" else row[key] for key in writer.fieldnames})
    write_json(output / "split_buffer_exclusions.json", discarded)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
              "dataset": str(args.dataset.resolve()), "manifest": manifest, "seed": args.seed,
              "pipeline_sha256": sha256(Path(__file__)),
              "manifest_sha256": sha256(manifest) if manifest else None,
              "samples_available": len(records), "samples_used": len(used), "split_target_ratios": {"train":0.8,"val":0.1,"test":0.1},
              "spatial_partition_before_sampling": True,
              "spatial_partition_counts_before_sampling": {key: len(value) for key,value in population.items()},
              "spatial_partition_region_counts_before_sampling": {key: dict(Counter(row["zone"] for row in value)) for key,value in population.items()},
              "spatial_partition_ratios_before_sampling": {key: len(value)/sum(map(len, population.values())) for key,value in population.items()},
              "sampling_method": ("per_split_random_by_zone_minimum_one_per_zone" if args.mode == "smoke-multiregional"
                                  else "per_split_random_legacy" if args.mode == "smoke" else "all_retained_spatial_partition_records"),
              "split_counts": {key: len(value) for key,value in splits.items()},
              "split_region_counts": {key: dict(Counter(row["zone"] for row in value)) for key,value in splits.items()},
              "split_ratios_actual": {key: len(value)/len(used) for key,value in splits.items()},
              "split_method": ("zone_stratified_10240m_groups_ordered_sectors_with_boundary_buffer" if multiregional
                               else "geographic_10240m_groups_ordered_sectors_with_boundary_buffer"),
              "split_buffer_m": args.buffer_m, "split_buffer_exclusions": len(discarded), "data_issues": issues,
              "tensorflow_version": tf.__version__, "keras_version": keras.__version__, "numpy_version": np.__version__,
              "available_gpus": [str(gpu) for gpu in tf.config.list_physical_devices("GPU")],
              "cpu_threads": args.threads, "training_status": "started", "epochs_requested": args.epochs,
              "limitations": ["MNDWI pseudo-ground-truth is not independent of spectral inputs",
                              "cloud/shadow-free status not verified for legacy",
                              "positive-patch water filter excludes hard all-background negatives",
                              "ImageNet domain differs from Sentinel-2 surface reflectance",
                              "final extended dataset remains pending GEE exports" if args.mode == "smoke" else "regional extrapolation needs independent validation"]}
    write_json(output / "config.json", {key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()})
    write_json(output / "report.json", report)
    try:
        model, transfer = build_model(args.freeze_encoder or args.mode in SMOKE_MODES, args.learning_rate)
        report.update({"input_shape": list(model.input_shape), "output_shape": list(model.output_shape),
                       "pretrained_transfer": transfer, "model_parameters": model.count_params(),
                       "trainable_parameters": sum(int(np.prod(v.shape)) for v in model.trainable_variables)})
        with (output / "model_summary.txt").open("w", encoding="utf-8") as handle:
            model.summary(print_fn=lambda line: handle.write(line + "\n"))
        report["gradient_check"] = gradient_check(model, splits["train"][0])
        fixed_x, fixed_y, _, _ = read_pair(splits["train"][0])
        initial_pred = model(fixed_x[None], training=False)
        initial_loss = float(model.loss(fixed_y[None], initial_pred).numpy())
        weights_before = [value.copy() for value in model.get_layer("water_probability").get_weights()]
        train_data = make_dataset(splits["train"], args.batch_size, True, args.seed)
        val_data = make_dataset(splits["val"], args.batch_size, False, args.seed)
        test_data = make_dataset(splits["test"], args.batch_size, False, args.seed)
        callbacks = [keras.callbacks.TerminateOnNaN(),
                     keras.callbacks.ModelCheckpoint(str(output / "model_best.keras"), monitor="val_loss", save_best_only=True),
                     keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                     keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
                     keras.callbacks.CSVLogger(str(output / "history.csv"))]
        history = model.fit(train_data, validation_data=val_data, epochs=args.epochs, callbacks=callbacks, verbose=2, shuffle=False)
        weights_after = model.get_layer("water_probability").get_weights()
        report["weights_updated"] = any(not np.array_equal(before, after) for before,after in zip(weights_before, weights_after))
        require(report["weights_updated"], "Pesos decoder no cambiaron tras fit.")
        report["fixed_train_example_loss_before"] = initial_loss
        report["fixed_train_example_loss_after"] = float(model.loss(fixed_y[None], model(fixed_x[None], training=False)).numpy())
        report["fixed_train_example_loss_decreased"] = report["fixed_train_example_loss_after"] < initial_loss
        report["history"] = {key: [float(value) for value in values] for key,values in history.history.items()}
        require(all(math.isfinite(value) for values in report["history"].values() for value in values), "Historial con NaN/Inf.")
        # Mejor checkpoint por VALIDACION; test se evalua una sola vez y no
        # participa en umbral0.5, arquitectura, early stopping ni seleccion.
        best = keras.models.load_model(output / "model_best.keras")
        test_result, probability_cache = evaluate_test_once(best, test_data, splits["test"])
        report.update(test_result)
        require(all(math.isfinite(value) for value in report["test_metrics"].values()), "Metricas de test no finitas.")
        report["test_evaluations"] = 1
        best.save(output / "model_final.keras")
        export_predictions(probability_cache, splits["test"], output)
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for key in ("loss", "val_loss"):
            axes[0].plot(np.arange(1,len(report["history"][key])+1), report["history"][key], marker="o", label=key)
        for key in ("iou_water", "val_iou_water", "f1_water", "val_f1_water"):
            axes[1].plot(np.arange(1,len(report["history"][key])+1), report["history"][key], marker="o", label=key)
        for axis, title in zip(axes, ["BCE+Dice", "Metricas contra pseudomascara"]):
            axis.set(title=title, xlabel="Epoca")
            axis.legend(fontsize=8)
            axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / "history.png", dpi=140)
        plt.close(figure)
        report["training_status"] = "completed"
        report["epochs_completed"] = len(history.epoch)
        report["elapsed_seconds"] = time.perf_counter()-start
        write_json(output / "report.json", report)
        return report
    except Exception as error:
        report["training_status"] = "failed"
        report["error"] = repr(error)
        report["elapsed_seconds"] = time.perf_counter()-start
        write_json(output / "report.json", report)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("smoke", "smoke-multiregional", "final"), default="final")
    result.add_argument("--dataset", type=Path, default=ROOT / "Dataset_UNet_Unificado")
    result.add_argument("--manifest", type=Path)
    result.add_argument("--output", type=Path, default=ROOT / "entrenamiento" / "run_final")
    result.add_argument("--epochs", type=int, default=50)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--smoke-samples", type=int, default=None,
                        help="Default12 en smoke legado;40 en smoke-multiregional (32/4/4).")
    result.add_argument("--freeze-encoder", action="store_true")
    result.add_argument("--allow-ambiguous-zeros", action="store_true")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--buffer-m", type=float, default=2560)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    return result


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps({key:report[key] for key in ("mode","samples_used","split_counts","test_metrics","weights_updated","elapsed_seconds")}, ensure_ascii=False, indent=2))
