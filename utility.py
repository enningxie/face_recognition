# tools
import face_recognition as fr
import pickle

PWD = "/home/enningxie/Pictures/"


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
