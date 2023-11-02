import custom_plots as cp
from skimage import feature
from skimage.color import rgb2gray
import numpy as np



def img_thresholding(img, threshold):
    img[img > threshold] = 255
    img[img < threshold] = 0
    return img

class test_params():
  """
      Function to test parameters of several computer vision methods over an image
  """
  def __init__(self,
              method,
              images,
              params,
              fixed_params = False,
              visualize = False):

      self.method = method
      self.images = images
      self.params = params
      self.fixed_params = fixed_params if fixed_params else {}
      self.visualize = visualize
      keys, values = zip(*self.params.items())
      self.params_comb = [dict(zip(keys, v)) for v in itertools.product(*values)] 
      
  
  def __getitem__(self, idx):
      segments = []
      contours = []
      blancs = []
      img = self.images[idx]
      titles = []
      for combination in self.params_comb:
          combination.update(self.fixed_params)
          res = self.method(img,**combination)
          segments.append(res)
          contours.append(segmentation.mark_boundaries(img, res))
          blancs.append(np.ones((10,1,3)))
          titles += ['',combination,'']
      if self.visualize:
          cp.custom_grids([element for tup in zip(segments,blancs,contours) for element in tup],len(self.params_comb),3,titles, axis='off', figsize=(5,10)).show()
      return segments, contours

def get_ORB(img1, img2):
  descriptor_extractor = feature.ORB(n_keypoints=200)
  descriptor_extractor.detect_and_extract(img1)
  keypoints1 = descriptor_extractor.keypoints
  descriptors1 = descriptor_extractor.descriptors

  descriptor_extractor.detect_and_extract(img2)
  keypoints2 = descriptor_extractor.keypoints
  descriptors2 = descriptor_extractor.descriptors

  matches12 = feature.match_descriptors(descriptors1, descriptors2, cross_check=True)

  return keypoints1, keypoints2, matches12

def get_multi_ORB(de, imgs):
  keypoints = []
  descriptors = []
  for img in imgs:
    if len(img.shape) != 2:
      img = rgb2gray(img)
    de.detect_and_extract(img)
    keypoints.append(de.keypoints)
    descriptors.append(de.descriptors)
  return np.array(keypoints), np.array(descriptors)
