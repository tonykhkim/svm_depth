import glob
import numpy as np
import joblib
import cv2
from skimage.io import imread   #이미지를 읽어 들인다.
from skimage.transform import resize  #이미지의 크기를 변경할 수 있다.
from skimage.feature import hog   #이미지의 영역별로 쪼개고 해당 영역내의 픽셀 기울기(gradient)의 히스토그램(histogram) 계산
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

object_images = glob.glob('/home/unicon4/svm/depth_colormap_object/*.jpg', recursive=True)

objects = []
for i in object_images:
    i=cv2.imread(i)
    img=cv2.resize(i,(64,64))
    #objects.append(cv2.imread(i))
    objects.append(img)


##이미지가 가진 색상 정보 등 여러가지 중요하지 않은 특징이 학습을 방해할 수 있다.
##그래서 얼굴 이미지의 중요한 특징만 추출할 필요가 있다.
##여기세 사용할 수 있는 대표적인 방법이 이미지의 기울기 히스토그램을 사용하는 것이다.

object_hogs = []    #히스토그램 이미지를 담을 배열
object_features = []   #히스토그램 디스크립터(descriptor)를 담을 배열

#for i in range(240):
  #hog_desc, hog_image = hog(object_images[i],orientations = 8, pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True, multichannel=True)   #히스토그램을 만들 방향은 8 방향으로 지정   #이미지를 16x16의 크기로 쪼갬   #히스토그램을 가시화한 이미지가 생성되도록 visualize를 True로 지정   #모두 컬러 이미지이므로 multichannel 역시 True로 지정
for i in range(219):
  hog_desc, hog_image = hog(objects[i],orientations = 8, pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True,channel_axis=True)  
  #디스크럽터와 가시화 이미지를 차례로 배열에 담는다.
  object_hogs.append(hog_image)
  object_features.append(hog_desc)
  
  ##부 그룹의 데이터가 될 이미지 준비


noobject_images = glob.glob('/home/unicon4/svm/depth_colormap_noobject/*.jpg', recursive=True)
no_objects = []
for i in noobject_images:
    i=cv2.imread(i)
    img=cv2.resize(i,(64,64))
  #no_objects.append(cv2.imread(noobject))
    no_objects.append(img)


##부 그룹의 특징 벡터 추출

no_object_hogs=[]
no_object_features=[]

for i in range(219):
  hog_desc,hog_image=hog(no_objects[i],orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True,channel_axis=True)
  no_object_hogs.append(hog_image)
  no_object_features.append(hog_desc)

#plot_images(2,10,no_object_hogs)

##학습을 위한 데이터 만들어 학습하기

X,y=[],[]

for featurex in object_features:
  X.append(featurex)
  y.append(1)

for featurey in no_object_features:
  X.append(featurey)
  y.append(0)
#print(X)
print(y)



polynomial_svm_clf=Pipeline([
    ("scaler",StandardScaler()),
    ("svm_clf",SVC(C=1,kernel='poly',degree=5,coef0=10.0))
])

polynomial_svm_clf.fit(X,y)
"""
print(polynomial_svm_clf.score(X,y))  #정확도 평가

#학습시킨 모델을 현재 경로에 svm_model.pkl파일로 저장한다.
joblib.dump(polynomial_svm_clf,'/content/drive/MyDrive/model/svm_model.pkl')

##학습된 모델로 훈련 데이터에 대한 예측을 수행하고 얻은 결과를 y_hat으로 저장해 출력

yhat=polynomial_svm_clf.predict(X)
print(yhat)

##새로운 데이터에 적용해 보기


test_images = glob.glob('/home/unicon4/svm/testdata/*.jpg', recursive=True)
tests = []
for test in test_images:
    #img=resize(object,(64,64))
    tests.append(cv2.imread(test))

#cv2_imshow(tests[40])
#len(tests)

#test_hogs = []
test_features = []
for i in range(39):
  hog_desc,hog_image = hog(tests[i],orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True,channel_axis=True)
  test_features.append(hog_desc)


##평가를 위한 데이터 만들어 학습하기

#X2=[]

#for feature in test_features:
  #X2.append(feature)
  
y2=[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0]    #정답
#len(y2)

test_result=polynomial_svm_clf.predict(test_features)
print(test_result)
print(polynomial_svm_clf.score(test_features,y2))

print(accuracy_score(y2, test_result))
print(recall_score(y2, test_result))
print(precision_score(y2, test_result))
print(f1_score(y2, test_result))
"""
