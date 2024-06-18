## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import pickle

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog

with open('/home/unicon4/svm/feature_vector/feature_vector.pkl','rb') as f:
    vect=pickle.load(f)
    
print(vect)
print(type(vect))
