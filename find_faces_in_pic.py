# normal
import face_recognition as fr
from PIL import Image


def main():
    image = fr.load_image_file("/home/enningxie/Pictures/azt01.jpg")
    loc = fr.face_locations(image)
    len_ = len(loc)
    top, right, bottom, left = loc[0]  # 上右下左,顺时针
    pil_image = image[top:bottom, left:right]  # 上下左右
    pil_image_ = Image.fromarray(pil_image)
    pil_image_.show()


if __name__ == '__main__':
    main()
