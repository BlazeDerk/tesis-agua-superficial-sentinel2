"""Comprueba gráficos con datos sintéticos temporales, nunca con resultados simulados del estudio."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from graficar_resultados import generate


class FigureTests(unittest.TestCase):
    def test_refuses_incomplete_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'evaluacion_metadatos.json').write_text(json.dumps({'evaluation_executed': False}), encoding='utf-8')
            with self.assertRaises(ValueError):
                generate(path)
            self.assertFalse((path / 'figuras').exists())

    def test_synthetic_tables_export_all_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'evaluacion_metadatos.json').write_text(json.dumps({
                'status': 'external_evaluation_completed', 'evaluation_executed': True,
                'inference_executed': True, 'test_fixture_only': True,
            }), encoding='utf-8')
            rows = [{'method': m, 'valid_pixels': 10, 'iou': .5, 'f1': 2/3,
                     'iou_defined': True, 'f1_defined': True} for m in ['unet', 'ndwi', 'mndwi']]
            with (path / 'metricas_globales.csv').open('w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            scenes = [{**r, 'aggregate_id': str(s), 'n_cases': 2} for s in [1, 2] for r in rows]
            with (path / 'metricas_por_escena.csv').open('w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=list(scenes[0])); w.writeheader(); w.writerows(scenes)
            generate(path)
            for extension in ['png', 'pdf', 'svg']:
                files = list((path / 'figuras').glob('*.' + extension))
                self.assertEqual(len(files), 2)
                self.assertTrue(all(f.stat().st_size > 1000 for f in files))


if __name__ == '__main__':
    unittest.main()
