# save pic per sec
import face_recognition
import cv2
import numpy as np
import pickle


# Open the input movie file
input_movie = cv2.VideoCapture("./Videos/1_01.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

def load_(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data


result_encoding_ = load_("stu_0.pkl")
result_encoding1_ = load_("stu_1.pkl")
result_encoding2_ = load_("stu_2.pkl")
result_encoding3_ = load_("stu_3.pkl")

known_faces_ = [
    result_encoding_,
    result_encoding1_,
    result_encoding2_,
    result_encoding3_
]


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    if frame_number % 25 == 0:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        dis = face_recognition.face_distance(face_encodings, result_encoding_)
        face_names = np.around(dis, decimals=2)
        for (top, right, bottom, left), name in zip(face_locations, face_names):


        # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, str(name), (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        image = frame[:, :, ::-1]
        cv2.imwrite('./images/stu0/stu0_' + str(int(frame_number / 25)) + '.jpg', image)
        print("Writing frame {} / {}".format(frame_number, length))
    else:
        continue



# All done!
input_movie.release()
# cv2.destroyAllWindows()