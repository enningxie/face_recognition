# (772, 2795, 851, 2716)
from utility import *
from PIL import Image
a = tuple((602, 2905, 921, 2606))  # stu_0

b = tuple((940, 1660, 1400, 1250))  # stu_2

c = tuple((1273, 2380, 1837, 1850))  # stu_1

d = tuple((498, 3213, 677, 3034))  # stu_3
def main():
    data = load('/home/enningxie/Documents/Codes/Face_recognition/data/_shot0014_png.pkl')
    image = data['pic_image']
    print(a[0])
    image = draw_rectangle(d, image)
    image = Image.fromarray(image)
    image.show()

    print('done.')




if __name__ == '__main__':
    main()