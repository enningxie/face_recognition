# compare the result with already exist

import numpy as np
from utility import *
import face_recognition as fr


stu_encodings = []
def main():
    for i in range(5):
        pwd = PWD + "0face/stu3_" + str(i) + ".jpg"
        stu_encoding = load_image_to_encoding(pwd)
        stu_encodings.append(stu_encoding)
    numpy_stu_encodings = np.asarray(stu_encodings)
    result_encoding = numpy_stu_encodings.mean(axis=0)
    print(result_encoding.shape)

    data = load('/home/enningxie/Documents/Codes/Face_recognition/data/stu_3.pkl')
    dis = fr.face_distance(stu_encodings, data)
    dis_ = fr.face_distance(stu_encodings, result_encoding)
    print(dis)
    print(dis_)


if __name__ == '__main__':
    main()