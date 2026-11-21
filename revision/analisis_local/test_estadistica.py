import unittest
import numpy as np
from analizar_escenas import metrics, paired_bootstrap, exact_permutations


class StatisticsTests(unittest.TestCase):
    def test_pooled_iou_differs_from_mean_of_scene_ratios(self):
        x=np.array([[1,0,0,0],[0,0,9,0]],dtype=np.int64)
        self.assertAlmostEqual(metrics(x.sum(axis=0))[0],.1)
        self.assertAlmostEqual(metrics(x).mean(axis=0)[0],.5)

    def test_paired_draws_keep_identical_methods_identical(self):
        scene=np.array([[5,1,2,50],[30,8,1,10],[3,0,2,70]],dtype=np.int64)
        counts=np.repeat(scene[:,None,:],3,axis=1)
        distributions,draws=paired_bootstrap(counts,reps=79,seed=42)
        for result in distributions.values():
            np.testing.assert_array_equal(result[:,0],result[:,1])
            np.testing.assert_array_equal(result[:,0],result[:,2])
        repeated,again=paired_bootstrap(counts,reps=79,seed=42)
        np.testing.assert_array_equal(draws,again)

    def test_single_scene_bootstrap_degenerates_to_point(self):
        counts=np.array([[[5,1,2,50],[4,0,3,51],[6,2,1,49]]],dtype=np.int64)
        distributions,_=paired_bootstrap(counts,reps=11,seed=7)
        for distribution in distributions.values():
            np.testing.assert_allclose(distribution,np.broadcast_to(metrics(counts)[0],distribution.shape))

    def test_exact_permutation_identical_methods_has_unit_p(self):
        scene=np.array([[5,1,2,50],[30,8,1,10]],dtype=np.int64)
        counts=np.repeat(scene[:,None,:],3,axis=1)
        rows=exact_permutations(counts)
        self.assertTrue(all(r['exact_two_sided_p']==1 for r in rows))
        self.assertTrue(all(r['observed_difference_pp']==0 for r in rows))


if __name__=='__main__':unittest.main()
