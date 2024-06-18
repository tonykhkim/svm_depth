import cv2
from PIL import Image

img1 = cv2.imread('/home/unicon4/svm/new_new_new_depth_colormap_object/depth_colormap_object0.jpg')
#output = img[160:410, 270:385]
cv2.imshow("img", img1)
#cv2.imshow("output", output)
print("color map(shape):",img1.shape)
print("color map(data type):",img1.dtype)
print("depth color map value of (324,274) pixel: ",img1[324,274])

img2 = Image.open("/home/unicon4/svm/new_new_new_depth_colormap_object/depth_colormap_object0.jpg")
print(img2.filename)
print(img2.format)
print(img2.mode)
print(img2.size)
cv2.waitKey()
cv2.destroyAllWindows()


