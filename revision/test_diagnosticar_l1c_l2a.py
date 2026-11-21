import tempfile,unittest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
import diagnosticar_l1c_l2a as diag

class DiagnosticTests(unittest.TestCase):
    def test_four_variants_share_validity_and_denominator_support(self):
        a=np.ones((5,2,2),dtype='float32');b=a.copy()
        a[1]=2;a[3]=1;a[4]=1
        b[1]=1;b[3]=2;b[4]=2
        b[:,1,1]=np.nan
        ref=np.array([[1,0],[1,0]],dtype='uint8');valid=np.array([[1,1],[0,1]],dtype='uint8')
        common,_,metrics,flips=diag.evaluate_pair(a,b,ref,valid)
        self.assertEqual(common.sum(),2)
        self.assertEqual({m['valid_pixels'] for m in metrics},{2})
        l1=next(m for m in metrics if m['level']=='l1c' and m['index']=='ndwi')
        l2=next(m for m in metrics if m['level']=='l2a' and m['index']=='ndwi')
        self.assertEqual((l1['tp'],l1['fp']),(1,1))
        self.assertEqual((l2['fn'],l2['tn']),(1,1))
        self.assertEqual(next(f for f in flips if f['index']=='ndwi' and f['support']=='all')['l1c_positive_l2a_negative'],2)
        b[:,0,0]=0
        common,_,metrics,_=diag.evaluate_pair(a,b,ref,valid)
        self.assertEqual(common.sum(),1)

    def test_zero_index_is_background(self):
        a=np.ones((5,1,1),dtype='float32')
        _,_,metrics,_=diag.evaluate_pair(a,a,np.ones((1,1)),np.ones((1,1)))
        self.assertTrue(all(m['fn']==1 and m['tp']==0 for m in metrics))

    def test_alignment_preserves_reference_grid_and_masks_nodata(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source.tif';ref=Path(temp)/'ref.tif'
            grid=from_origin(500000,1000000,10,10)
            profile=dict(driver='GTiff',height=3,width=3,crs='EPSG:32617',transform=grid,dtype='uint16')
            a=np.ones((6,3,3),dtype='uint16');a[:,0,0]=65535
            with rasterio.open(source,'w',count=6,nodata=65535,**profile) as d:d.write(a)
            with rasterio.open(ref,'w',count=1,**profile) as d:d.write(np.zeros((1,3,3),dtype='uint16'))
            with rasterio.open(source) as src,rasterio.open(ref) as target,diag.aligned_l1c(src,target) as view:
                self.assertEqual(view.transform,target.transform)
                self.assertTrue(view.read(masked=True).mask[:,0,0].all())
            with rasterio.open(ref) as bad,rasterio.open(ref) as target:
                with self.assertRaises(ValueError):diag.aligned_l1c(bad,target)

if __name__=='__main__':unittest.main()
