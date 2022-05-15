import cv2 
import json
from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import pickle as pkl
import numpy

from PIL import Image
from matplotlib import pyplot as plt

import pandas as pd

with open('data/annotations.json') as f:
    d = json.load(f)
    df1 = pd.DataFrame(d['annotations'])
    df = pd.DataFrame(d['images'])



    annotation = df1.where(df1['image_id']== 105)
    annotation = annotation.dropna()
  
    bboxes = annotation.iloc[8]
    bboxes = np.array([bboxes['bbox']])
    image = df.loc[df['id'] == 105]
    #print(image[image['file_name']])
    
    file_name =image._get_value(105,'file_name')
    

    print(bboxes)
    img = cv2.imread('data/'+file_name)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    plotted_img = draw_rect(img, bboxes)
    plt.imshow(plotted_img)
    plt.show()

    

    
"""    
    
    for i in range(0,10000):
        print(d['images'][105]['file_name'])
        print(type(d['annotations'][i]['bbox'][0]))
        bboxes = [d['annotations'][344]['bbox']]
        image_id = d['annotations'][i]['image_id']
        where(d['annotations'][i]['id'] == 4)
        img = cv2.imread('data/'+d['images'][i]['file_name'])[:,:,::-1]
        #image.show()
        plotted_img = draw_rect(img, np.array(bboxes))
        plt.imshow(plotted_img)
        plt.show()
"""

