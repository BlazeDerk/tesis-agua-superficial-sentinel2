"""Load the frozen original checkpoint and infer on optical tiles, without training.

Importing this module does not import TensorFlow or download weights. The caller
is responsible for matching the optical DN radiometry and for restoring NoData
in output rasters. Any filler used for invalid input pixels must be recorded and
the affected support handled explicitly by the external evaluation protocol.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


CHECKPOINT_SHA256 = "9e28f92b557bdd2e9a2e7b068a6622254ddbf80080926b8c004368ad5683dda4"
CHECKPOINT_SIZE_BYTES = 199_255_390
PATCH_SIZE = 256
INPUT_BANDS = ("B2", "B3", "B4", "B8")
INPUT_NAME = "sentinel_B2_B3_B4_B8_div10000"
OUTPUT_NAME = "water_probability"
MODEL_NAME = "UNet_VGG16_Sentinel2_4band"
MODEL_PARAMETERS = 16_588_945
MEANS_BGR_NIR = (103.939, 116.779, 123.68, (103.939 + 116.779 + 123.68) / 3)


def verify_checkpoint(checkpoint: str | Path) -> dict:
    """Require the selected epoch-50 file, before any deserialization occurs."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint}")
    size = checkpoint.stat().st_size
    if size != CHECKPOINT_SIZE_BYTES:
        raise ValueError(
            f"Tamaño de checkpoint incorrecto: {size}; se esperaba "
            f"{CHECKPOINT_SIZE_BYTES} bytes del model_best.keras original."
        )
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != CHECKPOINT_SHA256:
        raise ValueError(
            f"SHA-256 de checkpoint incorrecto: {actual}. Se exige "
            f"{CHECKPOINT_SHA256}; no sustituir por otro modelo."
        )
    return {
        "path": str(checkpoint.resolve()),
        "bytes": size,
        "sha256": actual,
        "expected_epoch": 50,
    }


def load_original_model(checkpoint: str | Path):
    """Load only the verified trained checkpoint, safely and without compiling.

    The saved graph has one custom inference layer. BCEDice and
    WaterConfusionMetric are compile-time objects and are not required with
    compile=False. No build_model(), ImageNet initialization, fitting or
    threshold calibration is performed.
    """
    verify_checkpoint(checkpoint)
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow no está disponible en este intérprete. Ejecuta la "
            "inferencia en un entorno compatible y registra sus versiones."
        ) from error

    # This operation exactly matches the original serialized layer's call().
    @tf.keras.utils.register_keras_serializable(package="Tesis")
    class VGG4Preprocess(tf.keras.layers.Layer):
        def call(self, inputs):
            return tf.cast(inputs, tf.float32) * 255.0 - tf.constant(
                MEANS_BGR_NIR, dtype=tf.float32
            )

    model = tf.keras.models.load_model(
        str(checkpoint),
        custom_objects={
            "VGG4Preprocess": VGG4Preprocess,
            "Tesis>VGG4Preprocess": VGG4Preprocess,
        },
        compile=False,
        safe_mode=True,
    )
    if tuple(model.input_shape) != (None, PATCH_SIZE, PATCH_SIZE, 4):
        raise ValueError(f"Entrada inesperada del checkpoint: {model.input_shape}")
    if tuple(model.output_shape) != (None, PATCH_SIZE, PATCH_SIZE, 1):
        raise ValueError(f"Salida inesperada del checkpoint: {model.output_shape}")
    if model.name != MODEL_NAME or model.count_params() != MODEL_PARAMETERS:
        raise ValueError("La arquitectura cargada no coincide con el modelo original.")
    if model.layers[0].name != INPUT_NAME or model.layers[-1].name != OUTPUT_NAME:
        raise ValueError("Las capas de entrada/salida no coinciden con el modelo original.")
    affine = model.get_layer("vgg_bgr_nir_affine")
    if not isinstance(affine, VGG4Preprocess):
        raise ValueError("La capa personalizada de preprocesamiento no coincide.")
    return model


def predict_dn_tiles(model, batch_dn: np.ndarray) -> np.ndarray:
    """Return probabilities for an NHWC uint16 batch in B2/B3/B4/B8 order.

    Input is DN scaled by 10 000, not reflectance that has already been divided.
    The caller chooses that radiometric route explicitly; dtype alone cannot
    establish whether a dataset is TOA or surface reflectance. Channel order is
    preserved. Values are neither clipped nor reordered, and 65 535 is rejected
    as the original pipeline's NoData marker.
    """
    if not isinstance(batch_dn, np.ndarray):
        raise TypeError("batch_dn debe ser un array NumPy.")
    if batch_dn.dtype != np.dtype("uint16"):
        raise TypeError(
            "Se requieren DN uint16. No pasar reflectancias float ni "
            "convertirlas a entero sin una decisión radiométrica documentada."
        )
    if batch_dn.ndim != 4 or batch_dn.shape[0] < 1 or batch_dn.shape[1:] != (PATCH_SIZE, PATCH_SIZE, 4):
        raise ValueError(
            "Se requiere un lote no vacío NHWC (N,256,256,4), "
            "en orden B2,B3,B4,B8."
        )
    if np.any(batch_dn == np.uint16(65_535)):
        raise ValueError("El lote contiene NoData 65535; preparar el soporte explícitamente.")
    reflectance = batch_dn.astype(np.float32) / np.float32(10_000.0)
    return predict_reflectance_tiles(model, reflectance)


def predict_reflectance_tiles(model, batch_reflectance: np.ndarray) -> np.ndarray:
    """Infer on physical reflectance float32 in B2/B3/B4/B8 NHWC order.

    Scale and offset must have been applied exactly once by the acquisition
    adapter, according to the asset metadata. This function performs no division
    by 10 000, clipping, offset or channel reordering. It preserves finite
    negative values and values above 1. The checkpoint's own VGG4Preprocess
    performs its affine transformation when the model is called.
    """
    if not isinstance(batch_reflectance, np.ndarray):
        raise TypeError("batch_reflectance debe ser un array NumPy.")
    if batch_reflectance.dtype != np.dtype("float32"):
        raise TypeError(
            "Se requiere reflectancia física float32; convertir los DN con "
            "su escala y offset documentados antes de llamar esta función."
        )
    if batch_reflectance.ndim != 4 or batch_reflectance.shape[0] < 1 or batch_reflectance.shape[1:] != (PATCH_SIZE, PATCH_SIZE, 4):
        raise ValueError(
            "Se requiere un lote no vacío NHWC (N,256,256,4), "
            "en orden B2,B3,B4,B8."
        )
    if not np.all(np.isfinite(batch_reflectance)):
        raise ValueError("La reflectancia contiene NaN o infinitos; preparar el soporte explícitamente.")
    prediction = model(batch_reflectance, training=False)
    if hasattr(prediction, "numpy"):
        prediction = prediction.numpy()
    probabilities = np.asarray(prediction)
    expected = (batch_reflectance.shape[0], PATCH_SIZE, PATCH_SIZE, 1)
    if probabilities.shape != expected:
        raise ValueError(f"Probabilidades con forma {probabilities.shape}; se esperaba {expected}.")
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise TypeError("La salida del modelo debe contener probabilidades flotantes.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("La salida contiene probabilidades no finitas.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("La salida contiene probabilidades fuera de [0,1].")
    return probabilities.astype(np.float32, copy=False)
