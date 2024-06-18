## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import joblib
import pickle

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RealsenseCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start streaming
        self.pipeline.start(config)


    def get_frame_stream(self):
    
        svm_model = joblib.load('/home/unicon4/svm/svm_model.pkl')
        
        with open('/home/unicon4/svm/X.pkl','rb') as f:
            feature_vector=pickle.load(f)
            
        hogdef = cv2.HOGDescriptor()
        #hogdef.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        feature_vector=np.array(feature_vector)
        hogdef.setSVMDetector(feature_vector)
        
        hogdaim  = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)
        hogdaim.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())
        
        mode = True  
        
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            #color_frame = frames.get_color_frame()
            if not depth_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            #color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            #depth_colormap_dim = depth_colormap.shape
            #color_colormap_dim = color_image.shape
            
            if mode:
            # default 디텍터로 보행자 검출 
                found, _ = hogdef.detectMultiScale(depth_colormap)
                for (x,y,w,h) in found:
                    cv2.rectangle(depth_colormap, (x,y), (x+w, y+h), (0,255,255))
                
            else:
            # daimler 디텍터로 보행자 검출 
                found, _ = hogdaim.detectMultiScale(depth_colormap)
                for (x,y,w,h) in found:
                    cv2.rectangle(depth_colormap, (x,y), (x+w, y+h), (0,255,0))
            cv2.putText(depth_colormap, 'Detector:%s'%('Default' if mode else 'Daimler'),(10,50 ), cv2.FONT_HERSHEY_DUPLEX,1, (0,255,0),1)

            

            #cv2.imshow('hog_image',hog_image)
            

            #pred_y = svm_model.predict(object_features)
            #hog_desc=np.reshape(hog_desc,(-1,1))
            #hog_desc=np.reshape(hog_desc,(1,-1))
            #pred_y = svm_model.predict(hog_desc)
            

            # Show images
            #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            #cv2.putText(depth_colormap,pred_y,cv2.FONT_HERSHEY_SIMPLEX,2(0,0,255),5)
            #print(pred_y)
            cv2.imshow('RealSense2',depth_colormap)
            cv2.waitKey(1)
        
    def release(self):
        self.pipeline.stop()
    """   
    def SVM(test_x,test_y):
        svm_model = joblib.load('/home/unicon4/svm/svm_model.pkl')
        #pred_y = svm_model.predict(test_x)
    """   
        

if __name__=="__main__":
    #svm_model = joblib.load('/home/unicon4/svm/svm_model.pkl')
    real=RealsenseCamera()
    real.get_frame_stream()
    #pred_y = svm_model.predict(test_x)
