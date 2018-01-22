import face_recognition as fr
from PIL import Image

top, right, bottom, left = tuple((602, 2905, 921, 2606))
image = fr.load_image_file("/home/enningxie/Pictures/pic/496.jpg")
image = image[top: bottom, left: right]
num_face = len(fr.face_locations(image))
print(num_face)
# image = Image.fromarray(image)
# image.show()