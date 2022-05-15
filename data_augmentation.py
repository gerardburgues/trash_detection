import cv2 
import json
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from data_aug.data_aug import *
from data_aug.bbox_util import *

IMAGE_ID = 100

with open('../TACO/data/annotations.json') as f:
    d = json.load(f)
    annotations = pd.DataFrame(d['annotations'])
    images = pd.DataFrame(d['images'])

    annotation = annotations.where(annotations['image_id'] == IMAGE_ID)
    annotation = annotation.dropna()

    [x,y,w,h] = annotation.iloc[1]['bbox']
    file_name = str(images._get_value(IMAGE_ID, 'file_name'))
    
    img = cv2.imread('../TACO/data/' + file_name)
    cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255,0,0), 5)
    plt.imshow(img)
    plt.show()
