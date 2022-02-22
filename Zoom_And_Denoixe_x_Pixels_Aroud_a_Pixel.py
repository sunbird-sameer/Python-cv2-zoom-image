import cv2

img = cv2.imread("download.jpg")

height = img.shape[0]
width = img.shape[1]

a = 50 # +/- pixels around the pixel zoomed

# X,Y Coordinates of zooming pixel
x = 500
y = 100

crop_img = img[y:y+a, x:x+a]
crop_img = cv2.resize(crop_img, (1050, 1610))
crop_img = cv2.fastNlMeansDenoisingColored(crop_img,None,10,10,5,100)

cv2.imshow("cropped", crop_img)
cv2.imwrite("cropped2.png", crop_img)
