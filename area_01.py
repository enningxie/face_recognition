# set 4 area
import cv2
from utility import *

stu0 = tuple((602, 2905, 901, 2606))  # stu_0

stu2 = tuple((940, 1660, 1400, 1250))  # stu_2

stu1 = tuple((1273, 2380, 1837, 1850))  # stu_1

stu3 = tuple((498, 3213, 677, 3034))  # stu_3


image = cv2.imread("/home/enningxie/Pictures/smplayer_screenshots/shot0004.png")

image = draw_rectangle(a, image)

image = draw_rectangle(b, image)

image = draw_rectangle(c, image)

image = draw_rectangle(d, image)

cv2.imwrite("/home/enningxie/Pictures/tmp/area_01_01.jpg", image)

