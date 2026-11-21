"""Pruebas con dos casos sintéticos TEMP y un modelo simulado, sin pesos reales."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine

import reevaluar_historicos as historical


class FakeModel:
    def __init__(self): self.seen=[]
    def __call__(self,x,training=False):
        if training:raise AssertionError('No se debe entrenar.')
        self.seen.append(x.copy())
        return np.where(x[:,:,:,1]>.2,np.float32(.5),np.float32(.49))[:,:,:,None]


class HistoricalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.raw=np.full((4,256,256),1000,dtype='uint16')
        self.raw[1]=4000;self.raw[3]=2000;self.raw[1,:,:128]=1000
        self.labels=np.zeros((256,256),dtype='uint16');self.labels[128:]=1
        self.base={'driver':'GTiff','height':256,'width':256,'crs':'EPSG:32717',
                   'transform':Affine(10,0,600000,0,-10,9700000),'dtype':'uint16','nodata':None}
        self.image=self.root/'imagen.tif';self.mask=self.root/'mascara.tif'
        with rasterio.open(self.image,'w',**self.base,count=4) as dst:dst.write(self.raw)
        with rasterio.open(self.mask,'w',**self.base,count=1) as dst:dst.write(self.labels,1)

    def tearDown(self):self.tmp.cleanup()

    def record(self,name,zone,exposed):
        record={'name':name,'zone':zone,'split':'test','spatial_group':'fixture_group',
                'previous_exposure':exposed}
        record.update(historical.read_pair(self.image,self.mask,record))
        return record

    def test_mock_inference_uses_physical_reflectance_once_and_preserves_support(self):
        records=[self.record('a.tif','el_oro',False),self.record('b.tif','lagos_imbabura',True)]
        model=FakeModel();out=self.root/'salida'
        rows,summary,probs,_=historical.evaluate_records(records,{'b.tif'},model,out,2)
        self.assertEqual(len(rows),4);self.assertEqual(len(probs),2)
        self.assertTrue(all(r['valid_pixels']==65536 for r in rows))
        np.testing.assert_array_equal(model.seen[0][0,:,:,0],np.full((256,256),.1,dtype='float32'))
        np.testing.assert_array_equal(model.seen[0][0,:,:,1],self.raw[1].astype('float32')/10000)
        for row in rows:
            self.assertEqual([row[k] for k in historical.COUNTS],[16384,16384,16384,16384])
            self.assertEqual(row['f1'],.5)
        for source in probs:
            with rasterio.open(source['path']) as ds:
                self.assertEqual(ds.transform,self.base['transform']);self.assertEqual(ds.crs.to_epsg(),32717)
                self.assertTrue((ds.read_masks(1)>0).all())
        empty=next(r for r in summary if r['cohort']=='without_previous_exposure' and r['zone']=='lagos_imbabura' and r['method']=='unet')
        self.assertEqual(empty['status'],'no_samples');self.assertIsNone(empty['f1']);self.assertEqual(empty['n_patches'],0)

    def test_ndwi_tie_zero_is_background_without_uint16_wraparound(self):
        raw=self.raw.copy();raw[1]=2000;raw[3]=2000
        self.assertFalse(historical.ndwi_prediction(raw).any())
        raw[1]=1;raw[3]=60000
        self.assertFalse(historical.ndwi_prediction(raw).any())

    def test_invalid_original_support_is_rejected(self):
        raw=self.raw.copy();raw[:,0,0]=65535
        with rasterio.open(self.image,'w',**self.base,count=4) as dst:dst.write(raw)
        with self.assertRaisesRegex(ValueError,'No se excluirán'):
            self.record('a.tif','el_oro',False)

    def test_altered_hash_or_manifest_disagreement_is_rejected(self):
        actual=historical.sha(self.image)
        with self.assertRaises(ValueError): historical.check_asset(self.image,'0'*64,actual)
        with self.assertRaises(ValueError): historical.check_asset(self.image,'0'*64,'0'*64)
        self.assertEqual(historical.check_asset(self.image,actual,actual),actual)

    def test_original_comparison_does_not_force_reproduction(self):
        old={'test_confusion':{'tp':8,'fp':1,'fn':2,'tn':20}}
        path=self.root/'report.json';path.write_text(json.dumps(old),encoding='utf-8')
        current={'cohort':'all_test','zone':'ALL','method':'unet','tp':7,'fp':2,'fn':3,'tn':19}
        current.update(historical.metric_values(current))
        rows,equal=historical.compare_original([current],path)
        self.assertFalse(equal)
        self.assertEqual(next(r for r in rows if r['quantity']=='tp')['new_minus_original'],-1)

    def test_validate_only_does_not_import_or_call_tensorflow_model(self):
        import argparse,sys
        args=argparse.Namespace(out_dir=self.root/'validacion',batch_size=4,dataset_root='not_read',
                                splits='not_read',manifest='not_read',exposure='not_read',checkpoint=None,validate_only=True)
        metadata={'n_test':59,'valid_pixels':3866624,'cohort_sizes':{'all_test':59,'without_previous_exposure':55}}
        before='tensorflow' in sys.modules
        with patch.object(historical,'validate_inputs',return_value=([],set(),metadata)):
            result=historical.run(args)
        self.assertEqual(before,'tensorflow' in sys.modules)
        self.assertFalse(result['inference_executed']);self.assertFalse(result['metric_outputs_created'])
        self.assertEqual([p.name for p in args.out_dir.iterdir()],['validacion_historica.json'])


if __name__=='__main__':unittest.main()
