#####
import face_recognition
import pickle


# load data from file
def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data