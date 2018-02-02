import face_recognition as fr
import numpy as np
from utils import load_image_to_encoding, save

stu_encodings = []
pwd_ = '/home/enningxie/Pictures/0face/stu15_'
save_pwd = '/home/enningxie/Documents/Codes/Face_recognition/data/stu_15.pkl'
for i in range(5):
    pwd = pwd_ + str(i) + ".jpg"
    stu_encoding = load_image_to_encoding(pwd)
    stu_encodings.append(stu_encoding)
    print(i)
numpy_stu_encodings = np.asarray(stu_encodings)
result_encoding = numpy_stu_encodings.mean(axis=0)
save(save_pwd, result_encoding)
print(result_encoding.shape)