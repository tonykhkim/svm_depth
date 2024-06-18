## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import glob
import joblib
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

test_features=[]

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

svm_model = joblib.load('/home/unicon4/svm/svm_model_featurevect.pkl')

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))


found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)


if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    


# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        #depth_colormap_dim = depth_colormap.shape
        #color_colormap_dim = color_image.shape

        """
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        """

        # Show images
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('RealSense1', images)
        #print('depth_colormap(shape):',depth_colormap.shape)        #(480,640,3)
        #print('depth_colormap(data type):',depth_colormap.dtype)    #uint8 --> 0~255
        
        #print('depth_image(shape):',depth_image.shape)              #(480,640)
        #print('depth_image(data type):',depth_image.dtype)          #uint16 --> 0~65535
                
        #print('color_image(shape):',color_image.shape)              #(480,640,3)
        #print('color_image(data type):',color_image.dtype)          #uint8 --> 0~255
        
        #depth_image=cv2.line(depth_image,(240,320),(240,320),(0,0,255),10)
        
        ##cv2.imshow('RealSense_depth_colormap',depth_colormap)
        #cv2.imshow('Realsense_depth_image',depth_image)
        
        #cv2.imshow('visual_image',color_image)
        
        #output=depth_colormap[140:410,270:400]   #original size
        
        output=depth_colormap[160:410,270:385]
        
        depth_colormap2=cv2.resize(output,(64,64))
        
        hog_desc,hog_image = hog(depth_colormap2,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True,multichannel=True)
        
        #hog_desc,hog_image = hog(depth_colormap2,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True,channel_axis=-1)
  
        #test_features.append(hog_desc)
        #test_result=svm_model.predict(test_features)  
        test_result=svm_model.predict([hog_desc])
        #test_result=svm_model.predict(hog_desc)
        print(hog_desc)
        cv2.imshow('RealSense_depth_colormap',output)
        cv2.imshow('RealSense_color_image',color_image)
        if test_result[-1] == 1:
          img = cv2.putText(depth_colormap,"object",(240,320),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),1,cv2.LINE_AA)
        
        if test_result[-1] == 0:
          img = cv2.putText(depth_colormap,"not object",(240,320),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),1,cv2.LINE_AA)
        
        cv2.imshow('Realsense_depth_colormap',img)
        print(test_result)
        
        
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
