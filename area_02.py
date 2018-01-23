# find face from area
from utility import *
import face_recognition as fr
import cv2

image = fr.load_image_file("/home/enningxie/Pictures/smplayer_screenshots/_shot0014.png")

top, right, bottom, left = a

face = fr.face_locations(image[top: bottom, left: right], model='cnn')

image = draw_rectangle(face[0], image[top: bottom, left: right])

# image = draw_rectangle(face[1], image)

cv2.imwrite("/home/enningxie/Pictures/tmp/area_01_02.jpg", image)

print(len(face))