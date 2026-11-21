"""Analytical and raster-contract tests; no trained model or real results.

All output rasters and mock predictions live in TemporaryDirectory and are
removed after each test. This suite never creates human annotations or metrics
for the research data.
"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine

import inferir_y_evaluar as external


class ArrayTests(unittest.TestCase):
    def arrays(self):
        optical = np.full((5, 2, 3), 0.2, dtype=np.float32)
        optical[1] = [[0.6, 0.2, 0.4], [0.3, 0.2, 0.3]]
        optical[3] = [[0.1, 0.4, 0.4], [0.1, -0.2, 0.1]]
        optical[4] = [[0.1, 0.5, 0.2], [-0.3, 0.1, 0.1]]
        reference = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        valid = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.uint8)
        probability = np.array([[0.5, 0.9, 0.1], [0.8, 0.8, 0.8]], dtype=np.float32)
        return optical, reference, valid, probability

    def test_exact_common_support_thresholds_and_confusions(self):
        rows, control, common = external.evaluate_arrays(*self.arrays())
        self.assertEqual(common.tolist(), [[True, True, True], [False, False, False]])
        by = {row["method"]: row for row in rows}
        self.assertEqual([by["unet"][key] for key in external.COUNT_KEYS], [1, 1, 1, 0])
        self.assertEqual([by["ndwi"][key] for key in external.COUNT_KEYS], [1, 0, 1, 1])
        self.assertEqual([by["mndwi"][key] for key in external.COUNT_KEYS], [2, 0, 0, 1])
        self.assertAlmostEqual(by["ndwi"]["iou"], 0.5)
        self.assertEqual(by["mndwi"]["f1"], 1.0)
        self.assertEqual(control["common_valid_pixels"], 3)
        self.assertEqual(sum(value for key, value in control.items() if key.startswith("sequential_")), 3)

    def test_probability_nan_excluded_for_all_methods(self):
        arrays = list(self.arrays())
        arrays[3][0, 0] = np.nan
        rows, control, _ = external.evaluate_arrays(*arrays)
        self.assertEqual({row["valid_pixels"] for row in rows}, {2})
        self.assertEqual(control["sequential_probability_nonfinite"], 1)

    def test_gdal_invalid_and_negative_reflectance(self):
        arrays = list(self.arrays())
        arrays[0][1, 1, 1] = 0.1
        arrays[0][3, 1, 1] = -0.05
        masks = np.ones((2, 3), dtype=bool)
        masks[0, 0] = False
        rows, control, common = external.evaluate_arrays(*arrays, optical_gdal=masks)
        self.assertTrue(common[1, 1])
        self.assertFalse(common[0, 0])
        self.assertEqual(control["negative_reflectance_pixels_not_automatically_excluded"], 2)
        self.assertEqual({row["valid_pixels"] for row in rows}, {3})

    def test_absent_water_preserves_undefined_flags(self):
        counts = dict(tp=0, fp=0, fn=0, tn=9)
        metrics = external.binary_metrics(counts)
        self.assertEqual(metrics["iou"], 0)
        self.assertFalse(metrics["iou_defined"])
        self.assertFalse(metrics["f1_defined"])
        self.assertTrue(metrics["reference_has_no_water"])

    def test_pooled_counts_not_mean_of_ratios(self):
        def row(scene, tp, fn):
            counts = dict(tp=tp, fp=0, fn=fn, tn=0)
            return {"scene_id": scene, "cluster_id": scene, "method": "unet",
                    "valid_pixels": tp + fn, **counts, **external.binary_metrics(counts)}
        results = external.aggregate([row("a", 1, 0), row("b", 0, 9)], "global")
        self.assertAlmostEqual(results[0]["iou"], 0.1)
        self.assertNotEqual(results[0]["iou"], 0.5)

    def test_ndwi_uint_dn_is_rejected_in_physical_reflectance_contract(self):
        arrays = list(self.arrays())
        arrays[0] = (arrays[0] * 10000).astype(np.uint16)
        with self.assertRaises(external.ValidationError):
            external.evaluate_arrays(*arrays)


class RasterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.rows = []
        self.create_case("case_a", 0)
        self.create_case("case_b", 256)
        self.manifest = self.root / "manifest.csv"
        self.save_manifest()

    def tearDown(self):
        self.directory.cleanup()

    def create_case(self, identifier, col):
        transform = Affine(10, 0, 500000 + col * 10, 0, -10, 1000000)
        profile = dict(driver="GTiff", width=256, height=256, crs="EPSG:32717", transform=transform)
        paths = {kind: self.root / f"{identifier}_{kind}.tif" for kind in ("optical", "reference", "validity")}
        optical = np.full((5, 256, 256), 0.1, dtype=np.float32)
        optical[1] = 0.3
        optical[:, 0, 0] = np.nan
        with rasterio.open(paths["optical"], "w", count=5, dtype="float32", nodata=np.nan, **profile) as out:
            out.write(optical)
            out.descriptions = external.BANDS
        reference = np.zeros((256, 256), dtype=np.uint8)
        reference[:, :128] = 1
        valid = np.ones((256, 256), dtype=np.uint8)
        valid[-1] = 0
        for kind, values in (("reference", reference), ("validity", valid)):
            with rasterio.open(paths[kind], "w", count=1, dtype="uint8", **profile) as out:
                out.write(values, 1)
        self.rows.append({"case_id": identifier, "scene_id": "scene_23", "cluster_id": "cluster_23",
                          "split": "test", "row_off": 0, "col_off": col, "height": 256, "width": 256,
                          "reference_origin": external.REFERENCE_ORIGIN, "l1c_source": "synthetic_L1C",
                          "l2a_product": "synthetic_L2A",
                          **{kind + "_path": path.name for kind, path in paths.items()},
                          **{kind + "_sha256": external.sha256(path) for kind, path in paths.items()}})

    def save_manifest(self):
        with self.manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=external.REQUIRED)
            writer.writeheader()
            writer.writerows(self.rows)
        source_item = self.root / "synthetic_l2a_item.json"
        external.write_json(source_item, {"fixture": "synthetic local test item"})
        scenes = []
        for scene_id in sorted({row["scene_id"] for row in self.rows}):
            row = next(row for row in self.rows if row["scene_id"] == scene_id)
            scenes.append({"scene_id": scene_id, "cluster_id": row["cluster_id"],
                           "l1c_source": row["l1c_source"], "l2a_product": row["l2a_product"],
                           "item_path": source_item.name, "item_sha256": external.sha256(source_item),
                           "mask_path": row["reference_path"], "mask_sha256": row["reference_sha256"],
                           "valid_path": row["validity_path"], "valid_sha256": row["validity_sha256"]})
        plan = {"split": "test", "reference_origin": external.REFERENCE_ORIGIN, "scenes": scenes,
                "cases": [{key: row[key] for key in ("case_id", "scene_id", "cluster_id", "row_off", "col_off", "height", "width")}
                          for row in self.rows]}
        plan_path = self.root / "plan_congelado.json"
        external.write_json(plan_path, plan)
        plan_hash = external.sha256(plan_path)
        for row in self.rows:
            external.write_json(self.root / (row["case_id"] + ".json"),
                                {"plan_sha256": plan_hash, "l2a_item_sha256": external.sha256(source_item),
                                 "source_bands": [{"band": band} for band in external.BANDS], "manifest": row})
        external.write_json(self.root / "estado_adquisicion.json",
                            {"complete": True, "expected_cases": len(self.rows), "expected_scenes": len(scenes),
                             "plan_sha256": plan_hash, "manifest": self.manifest.name,
                             "manifest_sha256": external.sha256(self.manifest)})

    def test_validate_only_does_not_load_model_or_create_metrics(self):
        out = self.root / "validation"
        with patch("modelo_original.load_original_model", side_effect=AssertionError("must not load")), \
                patch("modelo_original.predict_reflectance_tiles", side_effect=AssertionError("must not predict")):
            metadata = external.run(self.manifest, None, out, validate_only=True)
        self.assertEqual(metadata["status"], "validated_inputs_only")
        self.assertFalse(metadata["inference_executed"])
        self.assertFalse(metadata["evaluation_executed"])
        self.assertEqual([p.name for p in out.iterdir()], ["validacion_paquete.json"])
        self.assertNotIn("tensorflow", sys.modules)

    def test_changed_input_hash_is_rejected(self):
        self.rows[0]["optical_sha256"] = "0" * 64
        self.save_manifest()
        with self.assertRaisesRegex(external.ValidationError, "SHA-256"):
            external.validate_manifest(self.manifest)

    def test_incomplete_acquisition_cannot_be_evaluated(self):
        state_path = self.root / "estado_adquisicion.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["complete"] = False
        external.write_json(state_path, state)
        with self.assertRaisesRegex(external.ValidationError, "incompleta"):
            external.run(self.manifest, None, self.root / "partial", validate_only=True)
        self.assertFalse((self.root / "partial").exists())

    def test_subset_of_frozen_plan_is_rejected(self):
        with self.manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=external.REQUIRED)
            writer.writeheader()
            writer.writerow(self.rows[0])
        state_path = self.root / "estado_adquisicion.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["manifest_sha256"] = external.sha256(self.manifest)
        external.write_json(state_path, state)
        with self.assertRaisesRegex(external.ValidationError, "exactamente todos"):
            external.validate_manifest(self.manifest)

    def test_case_receipt_must_match_frozen_plan(self):
        receipt_path = self.root / (self.rows[0]["case_id"] + ".json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["plan_sha256"] = "0" * 64
        external.write_json(receipt_path, receipt)
        with self.assertRaisesRegex(external.ValidationError, "otra versión del plan"):
            external.validate_manifest(self.manifest)

    def test_misaligned_reference_is_rejected(self):
        path = self.root / self.rows[0]["reference_path"]
        with rasterio.open(path, "r+") as dataset:
            prior = dataset.transform
            dataset.transform = Affine(prior.a, prior.b, prior.c + prior.a, prior.d, prior.e, prior.f)
        self.rows[0]["reference_sha256"] = external.sha256(path)
        self.save_manifest()
        with self.assertRaisesRegex(external.ValidationError, "Cuadrícula"):
            external.validate_manifest(self.manifest)

    def test_duplicate_window_is_rejected(self):
        self.create_case("case_c", 0)
        self.save_manifest()
        with self.assertRaisesRegex(external.ValidationError, "solapadas"):
            external.validate_manifest(self.manifest)

    def test_empty_support_is_not_reported_as_real_evaluation(self):
        for row in self.rows:
            path = self.root / row["validity_path"]
            with rasterio.open(path, "r+") as dataset:
                dataset.write(np.zeros((256, 256), dtype=np.uint8), 1)
            row["validity_sha256"] = external.sha256(path)
        self.save_manifest()
        out = self.root / "invalid"
        with self.assertRaisesRegex(external.ValidationError, "carece de soporte"):
            external.run(self.manifest, None, out, validate_only=True)
        self.assertFalse(out.exists())

    def test_mock_inference_preserves_probability_grid_and_paired_support(self):
        calls = []
        def predictor(model, batch):
            self.assertTrue(np.isfinite(batch).all())
            self.assertEqual(batch.dtype, np.dtype("float32"))
            self.assertTrue((batch[:, 0, 0] == 0).all())
            calls.append(batch.copy())
            return np.full((len(batch), 256, 256, 1), 0.5, dtype=np.float32)
        fake_checkpoint = self.root / "synthetic_checkpoint.keras"
        fake_checkpoint.write_bytes(b"synthetic fixture - not research weights")
        checkpoint_record = {"path": str(fake_checkpoint), "sha256": external.sha256(fake_checkpoint)}
        out = self.root / "mock_run"
        with patch("modelo_original.verify_checkpoint", return_value=checkpoint_record), \
                patch("modelo_original.load_original_model", return_value=object()), \
                patch("modelo_original.predict_reflectance_tiles", side_effect=predictor):
            metadata = external.run(self.manifest, fake_checkpoint, out, batch_size=2)
        self.assertEqual(len(calls), 1)
        self.assertTrue(metadata["evaluation_executed"])
        expected = 2 * (256 * 255 - 1)
        self.assertEqual(metadata["common_valid_pixels"], expected)
        with (out / "metricas_globales.csv").open(encoding="utf-8-sig", newline="") as handle:
            global_rows = list(csv.DictReader(handle))
        self.assertEqual({int(row["valid_pixels"]) for row in global_rows}, {expected})
        self.assertEqual(len(global_rows), 3)
        for row in self.rows:
            probability = out / "probabilidades" / (row["case_id"] + "_probabilidad.tif")
            with rasterio.open(probability) as dataset, rasterio.open(self.root / row["optical_path"]) as optical:
                self.assertEqual(dataset.crs, optical.crs)
                self.assertEqual(dataset.transform, optical.transform)
                self.assertEqual(dataset.nodata, -9999)
                self.assertEqual(dataset.read(1)[0, 0], -9999)
                self.assertEqual(dataset.read(1)[0, 1], 0.5)

    def test_existing_output_is_never_overwritten(self):
        out = self.root / "existing"
        out.mkdir()
        marker = out / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(external.ValidationError, "ya existe"):
            external.run(self.manifest, None, out, validate_only=True)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
