import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def process_data(data_path,file_path):
    df = pd.read_csv(file_path,sep=' ',header=None)
    names = df[0].values
    images = [os.path.join(data_path,f'images/{name}.jpg')  for name in names]
    masks = [os.path.join(data_path,f'annotations/trimaps/{name}.png')  for name in names]
    return images,masks

def load_data(path):
    train_valid_path = os.path.join(path,"annotations/trainval.txt")
    test_path = os.path.join(path,"annotations/test.txt")
    train_x,train_y = process_data(path,train_valid_path)
    test_x,test_y = process_data(path,test_path)
    train_x,valid_x = train_test_split(train_x,test_size=0.2,random_state=42)
    train_y,valid_y = train_test_split(train_y,test_size=0.2,random_state=42)
    return (train_x,train_y),(valid_x,valid_y),(test_x,test_y)


if __name__=="__main__":
    path = "oxford-iiit-pet/"
    (train_x,train_y),(valid_x,valid_y),(test_x,test_y) = load_data(path)
    print(f"Fataset: Train: {len(train_x)}, Test: {len(test_x)}, Validation: {len(valid_x)}")
