import unittest
import numpy as np
from geoai_gid_yrb.labels import GID15_PALETTE,rgb_to_idx15,remap_15_to_6
from geoai_gid_yrb.metrics import segmentation_metrics
from geoai_gid_yrb.normalization import compute_band_stats,normalize_bands
from geoai_gid_yrb.resolution import block_average,block_mode

class CoreTests(unittest.TestCase):
    def test_palette_roundtrip(self):
        palette={i:c for c,i in GID15_PALETTE}; idx=np.array([[1,5],[9,13]],np.uint8); rgb=np.array([[palette[int(v)] for v in row] for row in idx],np.uint8)
        np.testing.assert_array_equal(rgb_to_idx15(rgb),idx)
    def test_remap(self):
        src=np.array([1,5,8,10,13,0]); np.testing.assert_array_equal(remap_15_to_6(src),np.array([6,1,2,3,5,0],np.uint8))
    def test_metrics(self):
        ref=np.array([1,1,2,2]); pred=np.array([1,2,2,2]); r=segmentation_metrics(ref,pred,3,0); self.assertAlmostEqual(r['overall_accuracy'],.75)
    def test_normalization(self):
        x=np.arange(2*3*4*4).reshape(2,3,4,4); mean,std=compute_band_stats(x); z=normalize_bands(x,mean,std); np.testing.assert_allclose(z.mean((0,2,3)),0,atol=1e-6)
    def test_resolution(self):
        y=np.array([[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]); np.testing.assert_array_equal(block_mode(y,2),np.array([[1,2],[3,4]])); self.assertEqual(block_average(y,2).shape,(2,2))
if __name__=='__main__': unittest.main()
