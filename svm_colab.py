import matplotlib.pyplot as plt
import numpy as np
import joblib


from skimage.io import imread   #이미지를 읽어 들인다.
from skimage.transform import resize  #이미지의 크기를 변경할 수 있다.
from skimage.feature import hog   #이미지의 영역별로 쪼개고 해당 영역내의 픽셀 기울기(gradient)의 히스토그램(histogram) 계산

# Data Exploration - GTI vehicle image dataset
object_images = glob.glob('/home/unicon4/svm/depth_colormap_object/*.jpg', recursive=True)
noobject_images = glob.glob('/home/unicon4/svm/depth_colormap_noobject/*.jpg', recursive=True)

object_images = [[]]
"""
def extract_image_region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    crop_image = image[BASE_LINE - BASE_LINE_ROI_WIDTH: BASE_LINE + BASE_LINE_ROI_WIDTH, 20: 60]

    return crop_image
BASE_LINE_ROI_WIDTH = 25
BASE_LINE=35
"""
for i in range(57):       #0부터 14까지 for문이 돈다.
  #file = url + 'img{0:02d}.png'.format(i+1)    #이미지 파일 이름을 만든다.   #.format(): 문자열 포맷팅 함수이며 문자열을 출력할 때 서식 지정자를 사용하여 출력하고자 하는 경우에 사용한다.
  file ='img{0:02d}.png'.format(i+1)
  img = imread(file)     #이미지를 읽는다.
  img=resize(img,(64,64))    #이미지의 크기를 변경한다.
  #img_crop=extract_image_region_of_interest(img)
  object_images.append(img)    #face_images에 담는다.
#plot_images(15,2,object_images)


object_hogs = []    #히스토그램 이미지를 담을 배열
object_features = []   #히스토그램 디스크립터(descriptor)를 담을 배열

for i in range(57):
  hog_desc, hog_image = hog(object_images[i],orientations = 8, pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True, multichannel=True)   #히스토그램을 만들 방향은 8 방향으로 지정   #이미지를 16x16의 크기로 쪼갬   #히스토그램을 가시화한 이미지가 생성되도록 visualize를 True로 지정   #모두 컬러 이미지이므로 multichannel 역시 True로 지정
  
  #디스크럽터와 가시화 이미지를 차례로 배열에 담는다.
  object_hogs.append(hog_image)
  object_features.append(hog_desc)
  
  #1차원 벡터이면서, 128x1의 이미지인 것을 눈으로 확인하기 쉽게 128x16의 크기로 변경한다.
#이 벡터가 각 이미지를 설명하는 특징 벡터라고 할 수 있다. 
fig=plt.figure()
#fig,ax=plt.subplots(3,19,figsize=(10,6))
fig,ax=plt.subplots(3,19,figsize=(20,20))
for i in range(3):
  for j in range(19):
    ax[i,j].imshow(resize(object_features[i*19+j],(128,16)))
    
    ##부 그룹의 데이터가 될 이미지 준비

url='https://drive.google.com/drive/folders/14uGCHXqkbRuSmOU_OacsPPP5MPfsl83u?usp=share_link'
%cd /content/drive/MyDrive/no_object/
no_object_images=[[]]

for i in range(20):
  file = 'img{0:02d}.png'.format(i+1)
  img = imread(file)
  img = resize(img,(64,64))
  no_object_images.append(img)

#plot_images(2,10,no_object_images)

##부 그룹의 특징 벡터 추출

no_object_hogs=[]
no_object_features=[]

for i in range(20):
  hog_desc,hog_image=hog(no_object_images[i],orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True,multichannel=True)
  no_object_hogs.append(hog_image)
  no_object_features.append(hog_desc)

plot_images(2,10,no_object_hogs)

fig = plt.figure()
fig,ax = plt.subplots(2,10,figsize=(10,6))
for i in range(2):
  for j in range(10):
    ax[i,j].imshow(resize(no_object_features[i*10+j],(128,16)))
    
    ##학습을 위한 데이터 만들어 학습하기

X,y=[],[]

for feature in object_images:
  X.append(feature)
  y.append(1)

for feature in no_object_images:
  X.append(feature)
  y.append(0)
print(X)
print(y)

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

polynomial_svm_clf=Pipeline([
    ("scaler",StandardScaler()),
    ("svm_clf",SVC(C=1,kernel='poly',degree=5,coef0=10.0))
])
polynomial_svm_clf.fit(X,y)
print(polynomial_svm_clf.score(X,y))  #정확도 평가

#학습시킨 모델을 현재 경로에 svm_model.pkl파일로 저장한다.
joblib.dump(polynomial_svm_clf,'/content/drive/MyDrive/model/svm_model.pkl')

##학습된 모델로 훈련 데이터에 대한 예측을 수행하고 얻은 결과를 y_hat으로 저장해 출력

yhat=polynomial_svm_clf.predict(X)
print(yhat)

##새로운 데이터에 적용해 보기

url = 'https://github.com/dknife/ML/raw/main/data/Proj2/test_data/'
%cd /content/drive/MyDrive/test_data
test_images=[]

for i in range(23):
  file = 'img{0:02d}.png'.format(i+1)
  img=imread(file)
  img=resize(img,(64,64))
  test_images.append(img)

plot_images(1,23,test_images)

test_features = []
for i in range(23):
  hog_desc,hog_image = hog(test_images[i],orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True,multichannel=True)
  test_features.append(hog_desc)

##평가를 위한 데이터 만들어 학습하기

X2=[]

for feature in test_features:
  X2.append(feature)
  
y2=[0,0,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]    #정답

test_result=polynomial_svm_clf.predict(test_features)
print(test_result)
print(polynomial_svm_clf.score(X2,y2))
