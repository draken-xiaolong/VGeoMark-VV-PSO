import unittest
import numpy as np
import shapefile
from psoqim.transforms import sdwt_virtual,isdwt_virtual,sdwt_virtual_rows
from psoqim.algorithm import EmbedConfig,embed_dataset,_extract_votes,_result_from_votes
from psoqim.io import ShapeDataset
from psoqim.watermark import Watermark,logistic_encrypt,logistic_decrypt,nc,bit_accuracy

class CoreTests(unittest.TestCase):
 def test_explicit_haar_equivalence(self):
  x=np.array([1.,5.,-2.,8.,3.]);v=np.empty(2*len(x));v[::2]=x;v[1::2]=(x+np.roll(x,-1))/2
  l,h=sdwt_virtual(x)
  np.testing.assert_allclose(l,(v[::2]+v[1::2])/np.sqrt(2));np.testing.assert_allclose(h,(v[::2]-v[1::2])/np.sqrt(2));np.testing.assert_allclose(isdwt_virtual(l,h),x)
  np.testing.assert_allclose(sdwt_virtual_rows(np.vstack([x,x]))[1],np.vstack([h,h]))
 def test_metric_distinction(self):
  a=np.array([1,1,0,0]);b=np.array([1,0,1,0]);self.assertEqual(nc(a,b),.5);self.assertEqual(bit_accuracy(a,np.ones(4)),.5);self.assertAlmostEqual(nc(a,np.ones(4)),2/np.sqrt(8))
 def test_permutation_and_missing_votes(self):
  a=np.array([[0,1],[1,0]],np.uint8);np.testing.assert_array_equal(logistic_decrypt(logistic_encrypt(a)),a)
  w=Watermark(a,logistic_encrypt(a));r=_result_from_votes(np.zeros(4,dtype=int),np.zeros(4,dtype=int),w,'none');self.assertEqual(r.empty_bins,4);self.assertEqual(r.votes_total,0)
 def test_closed_multipart_roundtrip(self):
  s=shapefile.Shape(5);s.parts=[0,5];s.points=[[120,30],[120.2,30],[120.2,30.1],[120,30.1],[120,30],[120.3,30.2],[120.4,30.2],[120.4,30.3],[120.3,30.3],[120.3,30.2]]
  d=ShapeDataset(5,[],[[]],[s]);a=np.array([[0,1],[1,0]],np.uint8);w=Watermark(a,logistic_encrypt(a));c=EmbedConfig(n_jobs=1,preserve_parts=True)
  e=embed_dataset(d,w,c);p=e.dataset.shapes[0].points
  self.assertEqual(p[0],p[4]);self.assertEqual(p[5],p[9]);self.assertEqual(list(e.dataset.shapes[0].parts),[0,5]);self.assertEqual(len(p),10)
  self.assertLessEqual(_extract_votes(e.dataset,w,c,e.transform)[2],8)
if __name__=='__main__':unittest.main()
