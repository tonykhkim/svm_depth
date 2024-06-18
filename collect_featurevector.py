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

count = 0
feature_vector=[]
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

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


        # Show images
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('RealSense1', images)
        """
        print('depth_colormap(shape):',depth_colormap.shape)        #(480,640,3)
        print('depth_colormap(data type):',depth_colormap.dtype)    #uint8 --> 0~255
        
        print('depth_image(shape):',depth_image.shape)              #(480,640)
        print('depth_image(data type):',depth_image.dtype)          #uint16 --> 0~65535
                
        print('color_image(shape):',color_image.shape)              #(480,640,3)
        print('color_image(data type):',color_image.dtype)          #uint8 --> 0~255
        """
        depth_image=cv2.line(depth_image,(240,320),(240,320),(0,0,255),10)
        
        cv2.imshow('RealSense_depth_colormap',depth_colormap)
        cv2.imshow('Realsense_depth_image',depth_image)
        
        cv2.imshow('visual_image',color_image)
        """
        print("color image value of (240,320) Pixel: ", color_image[240,320])
        print("depth image value of (240,320) Pixel: ", depth_image[240,320])
        print("deotg color map value of (240,320) Pixel: ", depth_colormap[240,320])
        """
               
        depth_colormap2=depth_colormap[160:410,270:385]
        depth_colormap3=cv2.resize(depth_colormap2,(64,64))
        cv2.imshow("resized colormap",depth_colormap3)
        hog_desc,hog_image = hog(depth_colormap3,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True,multichannel=True)
        #print(hog_desc)
        
        
        if cv2.waitKey(1) == ord('c'):
            
            print(hog_desc)
            print(type(hog_desc))
            feature_vector.append(hog_desc)
            #a.tolist()
            count += 1
         

        elif cv2.waitKey(1) == 27:
            with open('/home/unicon4/svm/feature_vector/feature_vector.pkl','wb') as f:
                pickle.dump(feature_vector,f)
            break
            
finally:

    # Stop streaming
    pipeline.stop()
    

