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
input_movie = cv2.VideoCapture("./Videos/1.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('./Videos/output14.avi', fourcc, 25, (4096, 2160))

a = tuple((602, 2905, 901, 2606))  # stu_0

b = tuple((940, 1660, 1400, 1250))  # stu_2

c = tuple((1273, 2380, 1837, 1850))  # stu_1

d = tuple((498, 3213, 677, 3034))  # stu_3

known_locs = [
    a,
    c,
    b,
    d
]
n = len(known_locs)
count_locs = np.zeros((n, n))

known_names = ['xz', 'xqq', 'azt', 'ck']




def load_(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data


# Load some sample pictures and learn how to recognize them.

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

    # face_names = []
    if frame_number < 500:
        for i, face_encoding in enumerate(face_encodings):
            # new func 2
            top, right, bottom, left = face_locations[i]
            for k, known_loc in enumerate(known_locs):
                top_, right_, bottom_, left_ = known_loc
                if top > top_ and right < right_ and bottom < bottom_ and left > left_:
                    dises = face_recognition.face_distance(known_faces_, face_encoding)
                    for j, dis in enumerate(dises):
                        if dis < 0.3:
                            count_locs[k][j] += 1

    # set name to know_locs
    if frame_number == 500:
        locs_names = []
        for index1, count_loc in enumerate(count_locs):
            index2 = np.asarray(count_loc).argmax()  # name's index
            locs_names.append(known_names[index2])


        # Label the results
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # new func 3
        if frame_number > 500:
            for k, known_loc in enumerate(known_locs):
                top_, right_, bottom_, left_ = known_loc
                if top > top_ and right < right_ and bottom < bottom_ and left > left_:
                    name = locs_names[k]
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

input_movie.release()
# cv2.destroyAllWindows()
