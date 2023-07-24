

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


image = cv2.imread("image.jpg",0)
h,w = image.shape
print(h,w)


#step 1 - flips the image vertically
image_copy = np.copy(image)
row = 0
for i in range(h-1,-1,-1):
  image_copy[row,:] = image[i,:]
  row+=1

cv2.imwrite("vertical_flips.jpg",image_copy)

plt.figure(figsize = (12, 9))
plt.axis('off')
plt.imshow(image_copy,cmap='gray', vmin=0, vmax=255)
plt.show()

#step 2 - flips the image horizontally
image_copy = np.copy(image)
col = 0
for i in range(w-1,-1,-1):
  image_copy[:,col] = image[:,i]
  col+=1

cv2.imwrite("horizontal_flips.jpg",image_copy)

plt.figure(figsize = (12, 9))
plt.axis('off')
plt.imshow(image_copy,cmap='gray', vmin=0, vmax=255)
plt.show()


#step 3 - rotates the image 90 degrees counterclockwise
img_zeros = np.zeros((w,h))
row = 0
for i in range(0,w):
  img_zeros[row,:] = image[:,-i]
  row+=1
img_counterclockwise = np.copy(img_zeros)
cv2.imwrite("image_counterclockwise.jpg",img_counterclockwise)

plt.figure(figsize = (9, 12))
plt.axis('off')
plt.imshow(img_counterclockwise,cmap='gray', vmin=0, vmax=255)
plt.show()


#step 4 -  rotates the image 90 degrees clockwise
img_zeros = np.zeros((w,h))
col = 0
for i in range(0,h-1):
  img_zeros[:,col] = image[-i,:]
  col+=1
img_clockwise = np.copy(img_zeros)
cv2.imwrite("image_clockwise.jpg",img_clockwise)

plt.figure(figsize = (9, 12))
plt.axis('off')
plt.imshow(img_clockwise,cmap='gray', vmin=0, vmax=255)
plt.show()


##step 5 - resizes the image to half by keeping the aspect ratio
rezied_image = image[0::2,0::2]
cv2.imwrite("rezied_image.jpg",rezied_image)
print(image.shape)
print(rezied_image.shape)

plt.figure(figsize = (12, 9))
plt.axis('off')
plt.imshow(rezied_image,cmap='gray', vmin=0, vmax=255)
plt.show()


#step 6 - applies negative transformation on the image
negative_image = np.ones((h,w))*255
negative_image = negative_image-image

cv2.imwrite("negative_image.jpg",negative_image)

plt.figure(figsize = (12, 9))
plt.axis('off')
plt.imshow(negative_image,cmap='gray', vmin=0, vmax=255)
plt.show()



#gamma = log(mid*255)/log(mean)
mid = 0.4
mean = np.mean(image)
gamma = math.log(mid*255)/math.log(mean)

#gamma correction
image_gamma = np.power(image, gamma).clip(0,255).astype(np.uint8)
cv2.imwrite("image_gamma.jpg",image_gamma)

plt.figure(figsize = (12, 9))
plt.axis('off')
plt.imshow(image_gamma,cmap='gray', vmin=0, vmax=255)
plt.show()


control_histogram = [*range(0,256)]
indexes = control_histogram.copy()

h,w = image.shape

for i in range(0,h):
  for j in range(0,w):
    pixel_value = image[i,j]
    control_histogram[pixel_value]+=1




index = 0
fig = plt.figure(figsize = (10, 7))
plt.plot(indexes, control_histogram)
plt.xlabel("0-255 pixel value")
plt.ylabel("Count")
plt.title("Histogram chart")

plt.show()