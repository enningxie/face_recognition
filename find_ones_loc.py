# loc
from utility import *
from PIL import Image
import numpy as np
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

    data = load('/home/enningxie/Documents/Codes/Face_recognition/data/_shot0014_png.pkl')
    image = data['pic_image']
    image_locs = data['locs']

    image_encodings = fr.face_encodings(image, image_locs)

    dises = fr.face_distance(image_encodings, result_encoding)

    index_min = np.asarray(dises).argmin()

    image = draw_rectangle(image_locs[index_min], image)

    image = Image.fromarray(image)

    image.show()
    print(image_locs[index_min])

    print('done.')







if __name__ == '__main__':
    main()