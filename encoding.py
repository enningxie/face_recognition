# deal with example stu's encoding
from utility import *
import numpy as np
import face_recognition as fr
import cv2

stu_encodings = []



def main():
    for i in range(5):
        pwd = PWD + "0face/stu0_" + str(i) + ".jpg"
        stu_encoding = load_image_to_encoding(pwd)
        stu_encodings.append(stu_encoding)
    numpy_stu_encodings = np.asarray(stu_encodings)
    result_encoding = numpy_stu_encodings.mean(axis=0)
    print(result_encoding.shape)

    # 1. 用result_encoding去5张图中比较dis（相似度）
    # dises = fr.face_distance(stu_encodings, result_encoding)
    #
    # for i, dis in enumerate(dises):
    #     print(i, ": ", dis)

    result = load('pic_04.pkl')
    pic_image = result['pic_image']
    locs = result['locs']
    print(len(locs))
    locs_encodings = fr.face_encodings(pic_image, locs)
    locs_dis = fr.face_distance(locs_encodings, result_encoding)
    print(locs_dis)
    face_names = np.around(locs_dis, decimals=2)
    print('-----------------------------------------')
    print(face_names)

    for (top, right, bottom, left), name in zip(locs, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(pic_image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(pic_image, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(pic_image, str(name), (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    image = pic_image[:, :, ::-1]
    cv2.imwrite(PWD + 'result_04.jpg', image)

    print('done.')


if __name__ == '__main__':
    main()