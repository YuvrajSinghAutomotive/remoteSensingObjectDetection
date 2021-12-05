import os
import numpy as np
import pandas as pd

def process_data(data_path,file_path):
    df = pd.read_csv(file_path,sep=' ',header=None)
    names = df[0].values()
    images = [os.path.join(data_path,f'images/{name}.jpg')  for name in names]
    masks = [os.path.join(data_path,f'annotations/trimaps/{name}.png')  for name in names]
    return images,masks

def load_data(path):
    train_valid_path = os.path.join(path,"annotations/trainval.txt")
    test_path = os.path.join(path,"annotations/test.txt")
    train_x,train_y = process_data(path,train_valid_path)
    test_x,test_y = process_data(path,test_path)

if __name__=="__main__":
    path = "oxford-iiit-pet/"
    load_data(path)