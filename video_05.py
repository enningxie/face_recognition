import face_recognition
import cv2
import numpy as np
import pickle
import api_01
from utils import save
import os

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture("/home/enningxie/Videos/1_01.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('/home/enningxie/Videos/output99.avi', fourcc, 25, (4096, 2160))


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
    #face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")
    #face_encodings = face_recognition.face_encodings(frame, face_locations)

    result = api_01.face_rec(frame, frame_number)
    # cv2.waitKey(1)
    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    # output_movie.write(frame)
    print(result)
    if len(result) == 16:
        break

# All done!
save('./result.pkl', result)
input_movie.release()
# cv2.destroyAllWindows()
