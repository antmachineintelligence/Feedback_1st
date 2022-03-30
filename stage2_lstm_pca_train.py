#!/usr/bin/env python
# coding: utf-8

import os
import gc
import copy
import time
import random
import string
import json
import pickle
import re
import math
from numba import jit

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# For data manipulation
import numpy as np
import pandas as pd
import lightgbm as lgb

# Utils
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from multiprocessing import Pool
from joblib import Parallel, delayed

from util import *

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level = logging.INFO,format = "%(asctime)s - %(message)s", datefmt="%m/%d %H:%M:%S")
logging.info("code begin!!!!!!!")

model_name = time.strftime('%d_%H_%M_%S_',time.localtime(time.time()))+str(random.randint(0,1000))
logging.info(f"====== model_name: {model_name} ======")

num_jobs = 60

data_path  = './data/'

train_df = pd.read_csv(data_path+'train.csv')
IDS = train_df.id.unique()

id2label = {0:'Lead', 1:'Position', 2:'Evidence', 3:'Claim', 4:'Concluding Statement',
             5:'Counterclaim', 6:'Rebuttal', 7:'blank'}
label2id = {v:k for k,v in id2label.items()}


df_wf = pickle.load(open('./data/data_6model_offline712_online704_ensemble.pkl','rb'))

# dic_txt_feat = pickle.load(open('./data/dic_txt_feat.pkl','rb'))
dic_off_map = df_wf[['id','offset_mapping']].set_index('id')['offset_mapping'].to_dict()
dic_txt = df_wf[['id','text']].set_index('id')['text'].to_dict()

def change_label(x):
    res1  = x[:,8:].sum(axis=1)
    res2 = np.zeros((len(res1), 8))
    
    label_map = {0:5, 1:3, 2:2, 3:1, 4:4, 5:6, 6:7, 7:0}
    for i in range(8):
        if i == 7:
            res2[:,i] = x[:,label_map[i]]
        else:
            res2[:,i] = x[:,[label_map[i], label_map[i]+7]].sum(axis=1)

    return res1, res2

preds1_5fold = {}
preds2_5fold = {}
for irow,row in df_wf.iterrows():
    t1, t2 = change_label(row.pred)
    preds1_5fold[row.id] = t1
    preds2_5fold[row.id] = t2
    

valid_pred = pickle.load(open('./data/recall_data.pkl','rb'))

kfold_ids = pickle.load(open('./data/kfold_ids.pkl','rb'))

logging.info(f'valid_pred num:{len(valid_pred)}')

preds2_5fold_type = {}
for k,t in preds2_5fold.items():
    preds2_5fold_type[k] = np.array(t).argmax(axis=-1)
    

@jit(nopython=True)
def feat_speedup(arr):
    r_max, r_min, r_sum = -1e5,1e5,0
    for x in arr:
        r_max = max(r_max, x)
        r_min = min(r_min, x)
        r_sum += x
    return r_max, r_min, r_sum, r_sum/len(arr)

np_lin = np.linspace(0,1,7)

@jit(nopython=True)
def sorted_quantile(array, q):
    n = len(array)
    index = (n - 1) * q
    left = int(index)
    fraction = index - left
    right = left
    right = right + int(fraction > 0)
    i, j = array[left], array[right]
    return i + (j - i) * fraction

def get_percentile(array):
    x = np.sort(array)
    n = len(x)-1
    return x[[int(n*t) for t in np_lin[1:-1]]]

def fun_get_feat(data_sub):
    df_feat = []
    repeat_key_set = set()
    for cache in tqdm(data_sub):
        id = cache[0]
        typ = cache[1]
        start, end = cache[2]
        prediction = cache[3]

        dic={'id': id, 'label':cache[5], 'label_rate':max(0,cache[6])}
        dic['class'] = label2id[typ]

        repeat_key = id+str(typ)+str(start)+str(end)
        if repeat_key in repeat_key_set:
            continue
        repeat_key_set.add(repeat_key)
        
        # 段内统计特征
        other_type = [t for t in range(8) if t != dic['class']]
        preds2_all = np.array(preds2_5fold[id])[:,label2id[typ]]
        # preds2 = preds2_all[max(0,start - 2):min(preds2_all.shape[0] - 1,end+3)]  
        preds2lstm = preds2_all[max(0,start - 2):min(preds2_all.shape[0] - 1,end+3)]  
        # preds1lstm = preds1_all[max(0,start - 2):min(preds2_all.shape[0] - 1,end+3)]  
        # preds2lstm = np.concatenate([preds2lstm,preds1lstm],1)      
        df_feat.append([id, preds2lstm, cache[5], np.mean(np.array(preds2_5fold[id])[max(0,start - 2):min(preds2_all.shape[0] - 1,end+3)], 0), label2id[typ]])
        
    save_path = './cache/'+'_'.join([cache[0],cache[1],str(cache[2])])+'.pkl'
    pickle.dump(df_feat, open(save_path, 'wb+'))
    return save_path
#     return df_feat
    

data_splits = np.array_split(valid_pred.values, num_jobs)
results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
    delayed(fun_get_feat)(data_sub) for data_sub in data_splits
)

logging.info(f"====== load pickle ======")
df_feat = []
for path in tqdm(results):
    df_feat.extend(pickle.load(open(path,'rb')))


print(df_feat[:5])


import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(8,16)
        self.lstm = nn.LSTM(1, 32, num_layers=1, bidirectional=True)
        self.fc0 = nn.Linear(64+16, 64+16)
        self.fc1 = nn.Linear(64+16, 1)
        self.fc2 = nn.Linear(64+16, 8)
        self.dropout = nn.Dropout(p=0.2)
        self.init_parameters()

    def forward(self, x, x2):
        x = x.unsqueeze(2)
        sequence_output_l = (x.permute(1, 0, 2))
        sequence_output_l, _ = self.lstm(sequence_output_l)
#         print(sequence_output_l.shape)
        sequence_output_l = sequence_output_l.permute(1, 0, 2)[:,-1,:]
        sequence_output_l = torch.cat([sequence_output_l,self.embedding(x2).squeeze(1) * 0],1)
        sequence_output_l = self.dropout(sequence_output_l)
        sequence_output_l = nn.ReLU()(self.fc0(sequence_output_l))
        output1 = nn.Sigmoid()(self.fc1(sequence_output_l))
        output2 = nn.Sigmoid()(self.fc2(sequence_output_l))
        return output1,output2

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

class dataset_train(data.Dataset):
    def __init__(self, x=None, x2=None, label1=None, label12=None):
        self.x = x
        self.x2 = x2
        self.label1 = label1
        self.label12 = label12

    def __getitem__(self, idx):
        return np.array(self.x[idx]), np.array(self.x2[idx]), self.label1[idx], self.label12[idx]

    def __len__(self):
        return len(self.x)
            
lstmmodels = []
idlist_lstm = []
X = []
X2 = []
y = []
y2 = []

def transform_score(x,n = 32):
    res = np.zeros(n+1)+0.01
    res_count = np.zeros(n+1)+0.01
    for i in range(len(x)):
        start_index = int(np.floor((i * n)/len(x)))
        end_index = int(np.ceil(((i + 1) * n)/len(x)))
        for index in range(start_index,end_index+1):
            res[index] += x[i]
            res_count[index] += 1
    res = res/res_count
    return res

for i in range(len(df_feat)):
    X.append(([-1] * 128 + df_feat[i][1].reshape(-1).tolist())[-128:])
    X2.append([df_feat[i][4]])
    y.append(df_feat[i][2])
    y2.append(df_feat[i][3])
    idlist_lstm.append(df_feat[i][0])
    
from sklearn.decomposition import PCA,TruncatedSVD
pcaX = []
for i in range(500000):
    pcaX.append(transform_score(df_feat[i][1].reshape(-1)))
print(pcaX[:2])
random.seed(827)
pca = PCA(n_components=8)
pca.fit_transform(pcaX)
pickle.dump(pca, open('./pcamodel.h5', 'wb'))

idlist_lstm_dict = {v:k for k,v in enumerate(set(idlist_lstm))}
for fold in range(5):
    model = LSTMModel().cuda()
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": 0.0001},
        ],
    )
    
    optimizer.zero_grad()
    loss_ce = nn.BCELoss()
    loss_mse = nn.MSELoss()
    train_x = []
    train_x2 = []
    train_y = []
    train_y2 = []
    val_x = []
    val_x2 = []
    val_y = []
    val_y2 = []
    for i in range(len(df_feat)):
        if idlist_lstm_dict[idlist_lstm[i]] % 5 == fold:
            val_x.append(X[i])
            val_x2.append(X2[i])
            val_y.append(y[i])
            val_y2.append(y2[i])
        else:
            train_x.append(X[i])
            train_x2.append(X2[i])
            train_y.append(y[i])
            train_y2.append(y2[i])
    print(fold,len(train_x),len(val_x))
    for ep in range(1):
        loss1 = []
        loss2 = []
        train_loader = dataset_train(train_x,train_x2,train_y,train_y2)
        train_loader = torch.utils.data.DataLoader(
            train_loader, batch_size=24, shuffle=True, num_workers=1, pin_memory=True, drop_last=True
        )
        val_loader = dataset_train(val_x,val_x2,val_y,val_y2)
        val_loader = torch.utils.data.DataLoader(
            val_loader, batch_size=24, shuffle=False, num_workers=1, pin_memory=True, drop_last=False
        )
        model.train()
        for input,input2,label1,label2 in train_loader:
            input = input.float().cuda()
            input2 = input2.long().cuda()
            label1 = label1.float().cuda()
            label2 = label2.float().cuda()
            
            output1,output2 = model(input,input2)
            loss = loss_ce(output1.view(-1),label1) + loss_mse(output2,label2)
            loss.backward()
            loss1.append(loss_ce(output1.view(-1),label1).cpu().detach().numpy())
            loss2.append(loss_mse(output2,label2).cpu().detach().numpy())
            optimizer.step()
            optimizer.zero_grad()
            if len(loss1) > 50000:
                break
        print(ep,np.mean(loss1),np.mean(loss2),output1,output2)
        loss1 = []
        loss2 = []
        model.eval()
        for input,input2,label1,label2 in val_loader:
            input = input.float().cuda()
            input2 = input2.long().cuda()
            label1 = label1.float().cuda()
            label2 = label2.float().cuda()
            
            output1,output2 = model(input,input2)
            loss1.append(loss_ce(output1.view(-1),label1).cpu().detach().numpy())
            loss2.append(loss_mse(output2,label2).cpu().detach().numpy())

            if len(loss1) > 5000:
                break
        print(ep,np.mean(loss1),np.mean(loss2),output1,output2)
        print(label1,label2)
    lstmmodels.append(model)

pickle.dump(idlist_lstm_dict, open('./result/lstmid', 'wb+'))
for i in range(5):
    torch.save(lstmmodels[i],('./result/lstmmodel' + str(i)+'.h5'))
