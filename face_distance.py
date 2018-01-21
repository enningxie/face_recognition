# get distances between test_face and known_faces
import face_recognition as fr


pwd_ck = "/home/enningxie/Pictures/ck01.jpg"
pwd_xz = "/home/enningxie/Pictures/xz01.jpg"
pwd_azt = "/home/enningxie/Pictures/azt01.jpg"
pwd_test = "/home/enningxie/Pictures/azt02.jpg"


def load_face_to_encoding(pwd):
    image = fr.load_image_file(pwd)
    encoding = fr.face_encodings(image)[0]
    return encoding


def main():
    encoding_ck = load_face_to_encoding(pwd_ck)
    encoding_xz = load_face_to_encoding(pwd_xz)
    encoding_azt = load_face_to_encoding(pwd_azt)
    encoding_test = load_face_to_encoding(pwd_test)

    encoding_known = [
        encoding_ck,
        encoding_xz,
        encoding_azt
    ]

    faces_distance = fr.face_distance(encoding_known, encoding_test)

    for i, face_distance in enumerate(faces_distance):
        print(i, " : ", face_distance)

    print("done.")


if __name__ == '__main__':
    main()
