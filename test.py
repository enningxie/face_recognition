# test thing
import face_recognition as fr
from PIL import Image
# load image
face_image = fr.load_image_file("/home/enningxie/Pictures/ck01.jpg")

facial_features = fr.face_landmarks(face_image)
print(facial_features)
# pil_image = Image.fromarray(face_image)
# pil_image.show()