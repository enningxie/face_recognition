import face_recognition
import cv2
import numpy as np
import pickle

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture("./Videos/1_01.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('./Videos/output11.avi', fourcc, 25, (4096, 2160))


def load(pwd):
    stu_image = face_recognition.load_image_file(pwd)
    stu_encoding = face_recognition.face_encodings(stu_image)[0]
    return stu_image, stu_encoding


def load_(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data


# Load some sample pictures and learn how to recognize them.
cd0 = "./images/stu0_"
cd1 = "./images/stu1_"
cd2 = "./images/stu2_"
cd3 = "./images/stu3_"

known_face = []
known_face_encoding = []

known_face1 = []
known_face_encoding1 = []

known_face2 = []
known_face_encoding2 = []

known_face3 = []
known_face_encoding3 = []

for i in range(5):
    stu_image, stu_encoding = load(cd0 + str(i) + ".jpg")
    known_face.append(stu_image)
    known_face_encoding.append(stu_encoding)

for i in range(5):
    stu_image, stu_encoding = load(cd1 + str(i) + ".jpg")
    known_face1.append(stu_image)
    known_face_encoding1.append(stu_encoding)

for i in range(5):
    stu_image, stu_encoding = load(cd2 + str(i) + ".jpg")
    known_face2.append(stu_image)
    known_face_encoding2.append(stu_encoding)

for i in range(5):
    stu_image, stu_encoding = load(cd3 + str(i) + ".jpg")
    known_face3.append(stu_image)
    known_face_encoding3.append(stu_encoding)


known_face_encoding_ = np.asarray(known_face_encoding, dtype=np.float32)
known_face_encoding1_ = np.asarray(known_face_encoding1, dtype=np.float32)
known_face_encoding2_ = np.asarray(known_face_encoding2, dtype=np.float32)
known_face_encoding3_ = np.asarray(known_face_encoding3, dtype=np.float32)
    # print(known_face_encoding_.mean(axis=0).shape)
    # print(known_face_encoding_[0].shape)

result_encoding = known_face_encoding_.mean(axis=0)
result_encoding1 = known_face_encoding1_.mean(axis=0)
result_encoding2 = known_face_encoding2_.mean(axis=0)
result_encoding3 = known_face_encoding3_.mean(axis=0)

known_faces = [
    result_encoding,
    result_encoding1,
    result_encoding2,
    result_encoding3
]

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

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.35)
        dises = face_recognition.face_distance(known_faces, face_encoding)
        dises_ = np.around(dises, decimals=2)
        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name = "xz_" + str(dises_[0])
        elif match[1]:
            name = "azt_" + str(dises_[1])
        elif match[2]:
            name = "ck_" + str(dises_[2])
        elif match[3]:
            name = "xqq_" + str(dises_[3])

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        if name:
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
# cv2.destroyAllWindows()
