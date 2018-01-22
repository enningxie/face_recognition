# get a stu's encoding from an area
import face_recognition as fr
import cv2
import numpy as np
import pickle
# from utility import *
top, right, bottom, left = tuple((940, 1660, 1400, 1250))
# Open the input movie file
input_movie = cv2.VideoCapture("./Videos/1.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

encodings = []
frame_number = 0
while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1
    # frame[top: bottom, left: right]
    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = fr.face_locations(frame[top: bottom, left: right])
    try:
        face_encoding = fr.face_encodings(frame[top: bottom, left: right], face_locations)[0]
        encodings.append(face_encoding)
    except:
        continue



    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))

print("length: ", len(encodings))
result = np.asarray(encodings).mean(axis=0)
# save("stu_02.pkl", result)
with open("stu_02.pkl", 'wb') as file:
    pickle.dump(result, file)
print("shape: ", result.shape)
# All done!
input_movie.release()
# cv2.destroyAllWindows()