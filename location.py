import face_recognition as fr
from utility import *


pic = '/home/enningxie/Pictures/smplayer_screenshots/shot0004.png'
# 2. 读入一张图片
pic_image = fr.load_image_file(pic)
# 找到图中所有的面部的location
locs = fr.face_locations(pic_image, number_of_times_to_upsample=0, model='cnn')

result = {
    'pic_image': pic_image,
    'locs': locs
}
save('/home/enningxie/Documents/Codes/Face_recognition/data/shot0004.pkl', result)

print('done.')