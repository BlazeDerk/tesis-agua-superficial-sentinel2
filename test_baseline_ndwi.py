"""Pruebas del contrato numérico/de procedencia, sin ejecutar ninguna red."""
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
import baseline_ndwi as b

np=b.np

class TestNDWI(unittest.TestCase):
    def test_uint16_subtraction_does_not_wrap(self):
        prediction, details=b.ndwi_arrays(np.array([1,60000],np.uint16),np.array([60000,1],np.uint16),np.ones(2,bool))
        np.testing.assert_array_equal(prediction,[False,True])
        self.assertLess(details['ndwi_min_on_original_valid'],0)
        self.assertGreater(details['ndwi_max_on_original_valid'],0)

    def test_zero_is_not_water(self):
        prediction,details=b.ndwi_arrays([1000],[1000],[True])
        self.assertFalse(prediction[0])
        self.assertEqual(details['ndwi_equal_threshold_on_original_valid'],1)

    def test_zero_denominator_fails_same_support_contract(self):
        prediction,details=b.ndwi_arrays([0],[0],[True])
        self.assertFalse(prediction[0])
        self.assertEqual(details['ndwi_undefined_on_original_valid'],1)
        with self.assertRaisesRegex(ValueError,'indefinido'):
            b.ensure_original_cohort(np.array([True]),details)

    def test_nodata_mask_cannot_silently_reduce_support(self):
        _,details=b.ndwi_arrays([100,100],[10,10],[True,False])
        with self.assertRaisesRegex(ValueError,'Validez distinta'):
            b.ensure_original_cohort(np.array([True,False]),details)

    def test_nonfinite_values_explicit(self):
        _,details=b.ndwi_arrays([np.nan],[10],[True])
        self.assertEqual(details['nonfinite_on_original_valid'],1)
        with self.assertRaises(ValueError):b.ensure_original_cohort(np.array([True]),details)

    def test_counts_and_metrics_known_example(self):
        counts=b.confusion([1,1,0,0],[1,0,1,0],[1,1,1,1])
        self.assertEqual([counts[k] for k in ['tp','fp','fn','tn']],[1,1,1,1])
        self.assertEqual(b.metric_values(counts)['iou'],1/3)
        self.assertEqual(b.metric_values(counts)['f1'],.5)

    def test_empty_water_metric_convention_explicit(self):
        counts=b.confusion([0,0],[0,0],[1,1]);m=b.metric_values(counts)
        self.assertEqual(m['iou'],0)
        self.assertFalse(m['iou_defined'])
        self.assertFalse(m['f1_defined'])

    def test_missing_or_wrong_hash_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'sample.tif';p.write_bytes(b'fixture')
            digest=b.sha256(p)
            with self.assertRaisesRegex(ValueError,'Falta SHA'):
                b.verified_hash(p,[('manifest',None),('splits',digest)])
            with self.assertRaisesRegex(ValueError,'discrepan'):
                b.verified_hash(p,[('manifest',digest),('splits','0'*64)])
            with self.assertRaisesRegex(ValueError,'discrepante'):
                b.verified_hash(p,[('manifest','0'*64),('splits','0'*64)])
            self.assertEqual(b.verified_hash(p,[('manifest',digest),('splits',digest)]),digest)

    def test_unknown_exposure_hard_failure(self):
        tests=[{'name':'test.tif','zone':'el_oro'}]
        rows=[{'final_name':'other.tif','zone':'el_oro','final_split':'test','prior_split':'test','prior_experiment':'smoke'}]
        with self.assertRaisesRegex(ValueError,'no pertenece'):
            b.validate_exposures(rows,tests,expected_count=1)

    def test_duplicate_exposure_hard_failure(self):
        tests=[{'name':'test.tif','zone':'el_oro'}]
        row={'final_name':'test.tif','zone':'el_oro','final_split':'test','prior_split':'test','prior_experiment':'smoke'}
        with self.assertRaisesRegex(ValueError,'duplicada'):
            b.validate_exposures([row,row],tests,expected_count=1)

    def test_missing_zone_not_zeroed_in_macro(self):
        rows=[]
        for name,zone in [('a','el_oro'),('b','lagos_imbabura')]:
            c=b.confusion([1],[1],[1]);rows.append({'name':name,'zone':zone,**c})
        summary=b.summarize(rows,{'b'})
        noex=[r for r in summary if r['cohort']=='without_previous_exposure']
        macro=next(r for r in noex if r['aggregation']=='unweighted_region_mean')
        self.assertEqual(macro['n_zones_present'],1)
        self.assertEqual(macro['iou'],1)
        imb=next(r for r in noex if r['zone']=='lagos_imbabura')
        self.assertEqual(imb['status'],'no_samples')
        self.assertIsNone(imb['iou'])

    def test_common_scale_leaves_ratio_and_prediction_unchanged(self):
        green=np.array([1000,2000,3000]);nir=np.array([100,2000,4000])
        prediction,_=b.ndwi_arrays(green,nir,np.ones(3,bool))
        scaled,_=b.ndwi_arrays(green/10000.,nir/10000.,np.ones(3,bool))
        np.testing.assert_array_equal(prediction,scaled)

if __name__=='__main__':unittest.main(verbosity=2)
