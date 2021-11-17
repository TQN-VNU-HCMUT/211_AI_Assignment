from cv2 import imread
from matplotlib.pyplot import imshow, show
from src.complete_model import CompleteModel

path = 'data/CMND/CMND-9.jpg'
model = CompleteModel()
img = imread(path)

detected_img, data = model.predict(img)
print(data)
imshow(detected_img)
show()
