# To use, add the file to the folder of images and change the directory address
from PIL import Image, ImageOps
import fnmatch, os

# change the directory
num_img = len(fnmatch.filter(os.listdir('/home/yanjun_yang/video dataset/Clip11-20/Clip11_1D'),'*.jpg'))

# print(img_size)
# old_size = 960, 540
# new_size = 1060, 640

i = 2
while i < num_img+1: 
	num = str(i)
	img = Image.open('Clip11_'+num+'D.jpg')
	img_size = img.size

	img_w_border = ImageOps.expand(img,border=50,fill='white')
	img_w_border.save('Clip11_'+num+'D.jpg')
	i = i+1
