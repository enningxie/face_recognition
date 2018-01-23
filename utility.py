# tools
import face_recognition as fr
import pickle
import cv2

PWD = "/home/enningxie/Pictures/"

a = tuple((602, 2905, 901, 2606))  # stu_0

b = tuple((940, 1660, 1400, 1250))  # stu_2

c = tuple((1273, 2380, 1837, 1850))  # stu_1

d = tuple((498, 3213, 677, 3034))  # stu_3


# give image_path return image_encoding
def load_image_to_encoding(pwd):
    image = fr.load_image_file(pwd)
    encoding = fr.face_encodings(image)[0]
    return encoding


# svae data to file
def save(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


# load data from file
def load(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data


# draw rectangle given a loc
def draw_rectangle(loc, image):
    top, right, bottom, left = loc
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    return image


