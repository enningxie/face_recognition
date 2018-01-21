import face_recognition as fr
from utility import *


pic = '/home/enningxie/Pictures/pic/607.jpg'
# 2. 读入一张图片
pic_image = fr.load_image_file(pic)
# 找到图中所有的面部的location
locs = fr.face_locations(pic_image, number_of_times_to_upsample=0, model='cnn')

result = {
    'pic_image': pic_image,
    'locs': locs
}
save('pic_04.pkl', result)

print('done.')