import os, shutil
import pickle 
import numpy as np 
from tqdm import tqdm

# first we go through the train files and copy them over to the new directory
train_path = 'dataset_INTERACTION/preprocess/train_interaction'
val_path = 'dataset_INTERACTION/preprocess/val_interaction'
train_all_path = 'dataset_INTERACTION/preprocess/train_all_interaction'

if not os.path.isdir(train_all_path):
    os.makedirs(train_all_path)  

print("Moving train files")
# copy train files over to train_all
src_files = os.listdir(train_path)
for file_name in tqdm(src_files):
    full_file_name = os.path.join(train_path, file_name)
    shutil.copy(full_file_name, train_all_path)

print("Moving val files")
# copy val files over to train_all but rename the indices and rename the indices
for i in tqdm(range(11794)):
    full_file_name = os.path.join(val_path, '{}.p'.format(i))
    data = np.load(os.path.join(val_path, "{}.p".format(i)), allow_pickle=True)
    data['idx'] += 47584

    f = open(os.path.join(train_all_path, "{}.p".format(data['idx'])), 'wb')
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

assert len(os.listdir(train_all_path)) == 47584 + 11794
    

