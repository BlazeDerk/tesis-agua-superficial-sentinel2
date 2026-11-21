import unittest
import numpy as np
from estadisticos_train_congelado import Moments

class MomentsTests(unittest.TestCase):
    def test_exact_population_moments_across_unequal_batches(self):
        m=Moments()
        a=np.array([[0,0],[1,1],[2,2],[-1,-1]],dtype='float32')
        b=np.array([[2,4,4],[3,5,5],[4,6,6],[1,3,3]],dtype='float32')
        m.add(a);m.add(b)
        raw=np.concatenate([a,b],axis=1).astype('float64')
        self.assertEqual(m.n,5)
        for i,r in enumerate(m.result()):
            self.assertAlmostEqual(r['mean'],raw[i].mean(),places=13)
            self.assertAlmostEqual(r['std_population'],raw[i].std(ddof=0),places=13)
        # ddof=1 must not silently substitute a sample deviation.
        self.assertNotEqual(m.result()[0]['std_population'],raw[0].std(ddof=1))

if __name__=='__main__':unittest.main()
