import numpy as np
import pickle
import os
from tqdm import tqdm

# first we create the mapping dictionary for train
train_dict = {}
for idx, name in tqdm(enumerate(os.listdir('dataset_AV2/train'))):
    train_dict[idx] = name 

print(len(train_dict))

with open('dataset_AV2/mapping_train_argoverse2.pkl', 'wb') as f:
    pickle.dump(train_dict, f)

# mapping dictionary for val
val_dict = {}
for idx, name in tqdm(enumerate(os.listdir('dataset_AV2/val'))):
    val_dict[idx] = name 

print(len(val_dict))

with open('dataset_AV2/mapping_val_argoverse2.pkl', 'wb') as f:
    pickle.dump(val_dict, f)
