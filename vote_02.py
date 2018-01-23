#
import face_recognition
import cv2
import numpy as np
import pickle
from utility import *

a = tuple((602, 2905, 901, 2606))  # stu_0

b = tuple((940, 1660, 1400, 1250))  # stu_2

c = tuple((1273, 2380, 1837, 1850))  # stu_1

d = tuple((498, 3213, 677, 3034))  # stu_3

known_locs = [
    a,
    c,
    b,
    d
]
n = len(known_locs)
count_locs = np.zeros((n, n))


def load_(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data


result_encoding_ = load_("/home/enningxie/Documents/Codes/Face_recognition/data/stu_0.pkl")
result_encoding1_ = load_("/home/enningxie/Documents/Codes/Face_recognition/data/stu_1.pkl")
result_encoding2_ = load_("/home/enningxie/Documents/Codes/Face_recognition/data/stu_2.pkl")
result_encoding3_ = load_("/home/enningxie/Documents/Codes/Face_recognition/data/stu_3.pkl")

known_faces_ = [
    result_encoding_,
    result_encoding1_,
    result_encoding2_,
    result_encoding3_
]




result = load('/home/enningxie/Documents/Codes/Face_recognition/data/shot0004.pkl')
image = result['pic_image']
face_locations = result['locs']

face_encodings = face_recognition.face_encodings(image, face_locations)

# for i, face_encoding in enumerate(face_encodings):
#     top, right, bottom, left = face_locations[i]
#     for k, known_loc in enumerate(known_locs):
#         top_, right_, bottom_, left_ = known_loc
#         if top > top_ & right < right_ & bottom < bottom_ & left > left_:
#             dises = face_recognition.face_distance(known_faces_, face_encoding)
#             for j, dis in enumerate(dises):
#                 if dis < 0.3:
#                     count_locs[k-1][j-1] += 1

for i, face_encoding in enumerate(face_encodings):
# new func
    dises = face_recognition.face_distance(known_faces_, face_encoding)
    for j, dis in enumerate(dises):
        if dis < 0.3:
            top, right, bottom, left = face_locations[i]
            for k, known_loc in enumerate(known_locs):
                top_, right_, bottom_, left_ = known_loc
                if top > top_ and right < right_ and bottom < bottom_ and left > left_:
                    print("hahah")
                    count_locs[k][j] += 1

print(count_locs)
print('done.')