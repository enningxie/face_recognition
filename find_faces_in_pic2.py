# cv
import face_recognition as fr
import cv2


def main():
    image = cv2.imread("/home/enningxie/Pictures/azt02.jpg")

    cv2.imshow("/home/enningxie/Pictures/cv2.jpg", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()