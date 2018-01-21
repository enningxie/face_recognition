# get one's encoding
import face_recognition as fr
pwd = "/home/enningxie/Pictures/"
image_1 = fr.load_image_file(pwd + "ck01.jpg")
encoding_1 = fr.face_encodings(image_1)[0]
image_2 = fr.load_image_file(pwd + "xz02.jpg")
encoding_2 = fr.face_encodings(image_2)[0]

dis = fr.face_distance([encoding_1], encoding_2)
print(dis)