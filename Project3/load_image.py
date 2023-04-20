import os
from PIL import Image

path = "C:/Users/franscho/Documents/CompSci-Projets/Project3/data/amptoph/no_split/"               
img = Image.open(os.path.join(path, 'amp/0001.png'))
print(img.size)