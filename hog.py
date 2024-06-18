## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################


import numpy as np
import cv2
import glob

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog

#Test

noobject_images = glob.glob('/home/unicon4/svm/new_new_new_depth_colormap_object/sample/*.jpg', recursive=True)
#notobject_images = glob.glob('/content/drive/MyDrive/depth_colormap_noobject/*.jpg', recursive=True)
no_objects = []
for k in noobject_images:
  img2=cv2.imread(k)
  output2=img2[160:410,270:385]
  image2=cv2.resize(output2,(64,64))
  no_objects.append(image2)
  #cv2.imshow(img2)
  #cv2.imshow(image2)


#Test
no_object_hogs=[]
no_object_features=[]

for l in range(3):
  hog_desc2,hog_image2=hog(no_objects[l],orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True,multichannel=True)
  no_object_hogs.append(hog_image2)
  no_object_features.append(hog_desc2)
  print(hog_desc2)
