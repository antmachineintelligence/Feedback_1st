import os
import pickle
import random
import numpy as np 
import pandas as pd 
from tqdm import tqdm

import torch
from torch import nn



data_path = 'feedback/'
print(os.listdir(data_path))
print('count of train:',len(os.listdir(data_path+'train')))
print('count of test:',len(os.listdir(data_path+'test')))
print('sample of train:',os.listdir(data_path+'train')[:5])


train_df = pd.read_csv(data_path+'train.csv')


from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
train_df['kfold'] = np.nan
for i,(train_index,test_index) in enumerate(gkf.split(train_df,groups=train_df.id)):
    train_df.loc[test_index,'kfold'] = i
print(train_df.groupby(['kfold'])['id'].count())
print(train_df.groupby(['kfold'])['id'].nunique())
train_df.to_csv(data_path+'train_folds.csv',index=False)
