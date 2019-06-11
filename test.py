import cv2
import os.path

filenames = "E:\\mask1" 
pathDir =  os.listdir(filenames)
n = 0
for allDir in pathDir:
	child = os.path.join('%s\%s' % (filenames, allDir))
	img = cv2.imread(child)
	
	n = n + 1;
	cv2.imwrite("E:\\mask\\" + "mask" + str(int(n / 10)) + str(n % 10) + ".jpg", img)  #store the image
