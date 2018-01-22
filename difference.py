# difference between new and old encoding
from utility import *
import face_recognition as fr

stu_encodings = []


def main():
    # func todo
    for i in range(5):
        pwd = PWD + "0face/stu3_" + str(i) + ".jpg"
        stu_encoding = load_image_to_encoding(pwd)
        stu_encodings.append(stu_encoding)
    numpy_stu_encodings = np.asarray(stu_encodings)
    result_encoding = numpy_stu_encodings.mean(axis=0)

    data = load('/home/enningxie/Documents/Codes/Face_recognition/data/stu_3.pkl')

    dis = fr.face_distance(stu_encodings, data)
    print(dis)

if __name__ == '__main__':
    main()