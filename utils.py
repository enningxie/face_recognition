#####
import face_recognition as fr
import pickle


# load data from file
def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data


# give image_path return image_encoding
def load_image_to_encoding(pwd):
    image = fr.load_image_file(pwd)
    encoding = fr.face_encodings(image)[0]
    return encoding


# svae data to file
def save(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print('save_op done.')