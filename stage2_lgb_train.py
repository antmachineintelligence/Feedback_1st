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

import pickle
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

from sklearn.decomposition import PCA,TruncatedSVD
pca = pickle.load(open('./result/pcamodel.h5', 'rb'))


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

def tuple_map(offset_mapping,threshold):
    paragraph_rk = []
    rk = 0
    last = 1
    for token_index in offset_mapping:
        if len(threshold) == 0:
            paragraph_rk.append(1)
        elif token_index[1] <= threshold[rk][1]:
            last = max(rk+1,last)
            paragraph_rk.append(last)
        else: 
            last = max(rk+2,last)
            paragraph_rk.append(last)
            if rk + 1 < len(threshold) - 1:
                rk += 1
            
    return paragraph_rk


def get_pos_feat(text, offset_mapping):

    paragraph_cnt = len(text.split('\n\n')) + 1

    paragraph_th = [m.span() for m in re.finditer('\n\n',text)]
    paragraph_rk = tuple_map(offset_mapping,paragraph_th)

    paragraph_rk_r = [paragraph_cnt-rk+1 if rk!=0 else 0 for rk in paragraph_rk]

    sentence_th = []
    for i,v in enumerate([m.span() for m in re.finditer('\n\n|\.|,|\?|\!',text)]):
        if i == 0:
            sentence_th.append(list(v))
        else:
            if v[0]==sentence_th[-1][-1]:
                sentence_th[-1][-1] = v[-1]
            else:
                sentence_th.append(list(v))
    sentence_cnt = len(sentence_th) + 1

    sentence_rk = tuple_map(offset_mapping,sentence_th)
    sentence_rk_r = [sentence_cnt-rk+1 if rk!=0 else 0 for rk in sentence_rk]

    last_garagraph_cnt = 0
    sentence_rk_of_paragraph = []
    for i in range(len(offset_mapping)):
        sentence_rk_of_paragraph.append(sentence_rk[i]-last_garagraph_cnt)
        if i+1 == len(offset_mapping) or paragraph_rk[i]!=paragraph_rk[i+1]:
            last_garagraph_cnt = sentence_rk[i]

    sentence_cnt_of_paragraph = []
    last_max = None
    for i in range(1,len(offset_mapping)+1):
        if i==1 or paragraph_rk[-i] != paragraph_rk[-i+1]:
            last_max = sentence_rk_of_paragraph[-i]
        sentence_cnt_of_paragraph.append(last_max)
    sentence_cnt_of_paragraph = sentence_cnt_of_paragraph[::-1]
 
    sentence_rk_r_of_paragraph = [s_cnt-rk+1 if rk!=0 else 0 for s_cnt,rk in zip(sentence_cnt_of_paragraph,sentence_rk_of_paragraph)]

    return paragraph_cnt,sentence_cnt,paragraph_rk,paragraph_rk_r,sentence_rk,sentence_rk_r, \
            sentence_cnt_of_paragraph,sentence_rk_of_paragraph,sentence_rk_r_of_paragraph

dic_txt_feat = {}
for row in df_wf:
    dic_txt_feat[row['id']] = get_pos_feat(row['text'], row['offset_mapping'])


def fun_get_feat(data_sub):
    df_feat = []
    df_feat2 = []
    for cache in tqdm(data_sub):
        id = cache[0]
        typ = cache[1]
        start, end = cache[2]
        prediction = cache[3]

        dic={'id': id, 'label':cache[5], 'label_rate':max(0,cache[6])}
        dic['class'] = label2id[typ]
        dic['post_flag'] = cache[4]
#         dic['cluster'] = dic_cluster[id]

        txt = dic_txt[id]

        txt_feat  = dic_txt_feat[id]
        dic['paragraph_cnt'] = txt_feat[0]
        dic['sentence_cnt'] = txt_feat[1]
        dic['paragraph_rk'] = txt_feat[2][start]
        dic['paragraph_rk_r'] = txt_feat[3][end]
        dic['sentence_rk'] = txt_feat[4][start]
        dic['sentence_rk_r'] = txt_feat[5][end]
        dic['sentence_cnt_of_paragraph'] = txt_feat[6][start]
        dic['sentence_cnt_of_paragraph2'] = txt_feat[6][end]
        dic['sentence_rk_of_paragraph'] = txt_feat[7][start]
        dic['sentence_rk_r_of_paragraph'] = txt_feat[8][end]
        dic['sub_paragraph_cnt'] = txt_feat[2][end] - txt_feat[2][start]
        dic['sub_sentence_cnt'] = txt_feat[4][end] - txt_feat[4][start]

        
        other_type = [t for t in range(8) if t != dic['class']]
        preds1_all = np.array(preds1_5fold[id])
        preds2_all = np.array(preds2_5fold[id])[:,label2id[typ]]
        preds4_all = np.array(preds2_5fold[id])[:,other_type].max(axis=-1)
        preds1 = preds1_all[start:end+1]
        preds2 = preds2_all[start:end+1]
        preds4 = preds4_all[start:end+1]
        preds2lstm = preds2_all[max(0,start - 2):min(preds2_all.shape[0] - 1,end+3)]  
        df_feat2.append([id,preds2lstm.reshape(-1),label2id[typ]])

        word_length = prediction[-1] - prediction[0] + 1
        token_length = len(dic_off_map[id])
        
        pca_res = pca.transform([transform_score(preds2lstm)])[0]
        for i in range(8):
            dic['pca_f'+str(i)] = pca_res[i] 
        pca_res = pca.transform([transform_score(preds1_all[max(0,start - 2):min(preds2_all.shape[0] - 1,end+3)])])[0]
        for i in range(8):
            dic['pca2_f'+str(i)] = pca_res[i] 
        dic['L1'] = word_length
        dic['L2'] = end - start + 1
        dic['text_char_length'] = len(txt)
        dic['text_word_length'] = len(txt.split())
        dic['text_token_length'] = token_length

        dic['word_start'] = prediction[0]
        dic['word_end'] = prediction[-1]
        dic['token_start'] = start
        dic['token_start2'] = start / token_length
        dic['token_end'] = end
        dic['token_end2'] = token_length - end
        dic['token_end3'] = end / token_length
        
        dic[f'head_preds1'] = preds1[0]
        dic[f'head2_preds1'] = preds1_all[start-1:start+2].sum()
        if len(preds1) > 1:
            dic[f'tail_preds1'] = preds1[-1]
            dic['max_preds1'], dic['min_preds1'], dic['sum_preds1'], dic['mean_preds1'] = feat_speedup(preds1[1:])
      
        sort_idx = preds1[1:].argsort()[::-1]
        tmp = []
        for i in range(5):
            if i < len(sort_idx):
                dic[f'other_preds1_{i}'] = preds1[1+sort_idx[i]]
                dic[f'other_preds1_idx_{i}'] = (1+sort_idx[i])/len(preds1)
                tmp.append(preds1[1+sort_idx[i]])
        if len(tmp):
            dic[f'other_preds1_mean'] = np.mean(tmp)

        dic[f'head_preds2'] = preds2[0]
        dic[f'tail_preds2'] = preds2[-1]
        dic['max_preds2'], dic['min_preds2'], dic['sum_preds2'], dic['mean_preds2'] = feat_speedup(preds2)

        dic[f'head_preds4'] = preds4[0]
        dic[f'tail_preds4'] = preds4[-1]
        dic['max_preds4'], dic['min_preds4'], dic['sum_preds4'], dic['mean_preds4'] = feat_speedup(preds4)

        sort_idx = preds2.argsort()
        tmp = []
        for i in range(5):
            if i < len(sort_idx):
                dic[f'other_preds2_{i}'] = preds2[sort_idx[i]]
                dic[f'other_preds2_idx_{i}'] = (sort_idx[i])/len(preds2)
                tmp.append(preds2[sort_idx[i]])
        if len(tmp):
            dic[f'other_preds2_mean'] = np.mean(tmp)

            
        for i,ntile in enumerate([sorted_quantile(preds2,i) for i in np_lin]):
            dic[f'preds2_trend{i}'] = ntile
        for i,ntile in enumerate(get_percentile(preds2)):
            dic[f'preds2_ntile{i}'] = ntile
        for i,ntile in enumerate([sorted_quantile(preds4,i) for i in np_lin]):
            dic[f'preds4_trend{i}'] = ntile
        for i,ntile in enumerate(get_percentile(preds4)):
            dic[f'preds4_ntile{i}'] = ntile
            
        for i in range(1,4):
            if start-i >= 0:
                dic[f'before_head2_prob{i}'] = preds2_all[start-i]
                dic[f'before_other_prob{i}'] = preds4_all[start-i]
                dic[f'before_other_type{i}'] = preds2_5fold_type[id][start-i]
                
            if end+i < len(preds1_all):
                dic[f'after_head2_prob{i}'] = preds2_all[end+i]
                dic[f'after_other_prob{i}'] = preds4_all[end+i]
                dic[f'after_other_type{i}'] = preds2_5fold_type[id][end+i]

        
        for mode in ['before', 'after']:
            for iw, extend_L in enumerate([math.ceil(word_length/2), word_length]):
                if mode == 'before':
                    if start-extend_L<0:
                        continue
                    preds1_extend = preds1_all[start-extend_L:start]
                    preds2_extend = preds2_all[start-extend_L:start]
                else:
                    if end+extend_L >=len(preds1_all):
                        continue
                    preds1_extend = preds1_all[end+1:end+extend_L]
                    preds2_extend = preds2_all[end+1:end+extend_L]
                    
                if len(preds1_extend) == 0:
                    continue
                dic[f'{mode}{iw}_head_preds1'] = preds1_extend[0]
                dic[f'{mode}{iw}_max_preds1'], dic[f'{mode}{iw}_min_preds1'], \
                dic[f'{mode}{iw}_sum_preds1'], dic[f'{mode}{iw}_mean_preds1'] = feat_speedup(preds1_extend)

                dic[f'{mode}{iw}_head_preds2'] = preds2_extend[0]
                dic[f'{mode}{iw}_max_preds2'], dic[f'{mode}{iw}_min_preds2'], \
                dic[f'{mode}{iw}_sum_preds2'], dic[f'{mode}{iw}_mean_preds2'] = feat_speedup(preds2_extend)

                dic[f'{mode}{iw}_sum_preds1_rate'] = dic[f'{mode}{iw}_sum_preds1'] / dic[f'sum_preds1']
                dic[f'{mode}{iw}_sum_preds2_rate'] = dic[f'{mode}{iw}_sum_preds2'] / dic[f'sum_preds2']
                dic[f'{mode}{iw}_max_preds1_rate'] = dic[f'{mode}{iw}_max_preds1'] / dic[f'max_preds1']
                dic[f'{mode}{iw}_max_preds2_rate'] = dic[f'{mode}{iw}_max_preds2'] / dic[f'max_preds2']

        df_feat.append(dic)

    save_path = './cache/'+'_'.join([cache[0],cache[1],str(cache[2])])+'.pkl'
    pickle.dump([df_feat,df_feat2], open(save_path, 'wb+'))
    return save_path
#     return df_feat
    



data_splits = np.array_split(valid_pred.values, num_jobs)
results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
    delayed(fun_get_feat)(data_sub) for data_sub in data_splits
)



logging.info(f"====== load pickle ======")
df_feat = []
df_feat2 = []
for path in tqdm(results):
    res1,res2 = pickle.load(open(path,'rb'))
    df_feat.extend(res1)
    df_feat2.extend(res2)



logging.info(f"====== lstm feat ======")

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
        

df_feat3 = [] 
idlist_lstm = []
submit = False
lstmmodels = []
X = []
X2 = []
y = []
y2 = []

for i in range(5):
    lstmmodels.append(torch.load('./result/lstmmodel' + str(i)+'.h5').cuda())
    lstmmodels[-1].eval()


idlist_lstm_dict = pickle.load(open('./result/lstmid', 'rb'))
for i in range(len(df_feat2)):
    X.append(([-1] * 128 + df_feat2[i][1].reshape(-1).tolist())[-128:])
    X2.append([df_feat2[i][2]])
    idlist_lstm.append(df_feat2[i][0])
del df_feat2

df_feat3 = np.zeros((len(idlist_lstm),9))
for fold in tqdm(range(5)):
    model = lstmmodels[fold]
    val_x = []
    val_x2 = []
    val_y = []
    val_y2 = []
    val_idx = []
    for i in range(len(idlist_lstm)):
        if idlist_lstm_dict[idlist_lstm[i]] % 5 == fold:
            val_x.append(X[i])
            val_x2.append(X2[i])
            val_y.append(0)
            val_y2.append(0)
            val_idx.append(i)


    val_loader = dataset_train(val_x,val_x2,val_y,val_y2)
    val_loader = torch.utils.data.DataLoader(
        val_loader, batch_size=64, shuffle=False, num_workers=1, pin_memory=True, drop_last=False
    )

    model.eval()
    pos = 0
    for input,input2,label1,label2 in val_loader:
        input = input.float().cuda()
        input2 = input2.long().cuda()
        
        output1,output2 = model(input,input2)
        df_feat3[val_idx[pos:pos + output1.shape[0]],:1] = output1.cpu().detach().numpy()
        df_feat3[val_idx[pos:pos + output1.shape[0]],1:] = output2.cpu().detach().numpy()
        pos += output1.shape[0]

del X,X2
del val_loader

df_feat = pd.DataFrame(df_feat)

print(df_feat.head(5))
df_feat3 = pd.DataFrame(df_feat3)
df_feat3.columns = ['lstmf' + str(i) for i in range(df_feat3.shape[1])]
print(df_feat.shape,df_feat3.shape)
df_feat = pd.concat([df_feat,df_feat3.iloc[:,:]],1)
print(df_feat3.head(5))
del df_feat3

inter_thresh = { 
    "Lead": 0.15,
    "Position": 0.15,
    "Evidence": 0.15,
    "Claim": 0.25,
    "Concluding Statement": 0.15,
    "Counterclaim": 0.25,
    "Rebuttal": 0.25,
}
group_dict = {}


logging.info(f"====== dataFrame ok ======")

params = {
          'boosting': 'gbdt',
          'objective': 'binary',
          'metric': {'auc'},
#           'objective': 'regression',
#           'metric': {'l2'},
          'num_leaves': 15,
          'min_data_in_leaf': 30,
          'max_depth': 5,
          'learning_rate': 0.03,
          "feature_fraction": 0.7,
          "bagging_fraction": 0.7,
          'min_data_in_bin':15,
#           "min_sum_hessian_in_leaf": 6,
          "lambda_l1": 5,
          'lambda_l2': 5,
          "random_state": 1996,
          "num_threads": num_jobs,
          }

valid_pred = df_feat[['id', 'class','word_start','word_end']].copy()
valid_pred['class'] = valid_pred['class'].map(lambda x:id2label[x])
valid_pred['lgb_prob'] = -1
for fold in range(5):
    df_feat_train = df_feat[df_feat.id.isin(kfold_ids[fold][0])].copy()
    df_feat_val = df_feat[df_feat.id.isin(kfold_ids[fold][1])].copy()

    lgb_train = lgb.Dataset(df_feat_train.drop(['id', 'label', 'label_rate'], axis=1), label=df_feat_train['label'])
    lgb_val = lgb.Dataset(df_feat_val.drop(['id', 'label', 'label_rate'], axis=1), label=df_feat_val['label'])

    clf = lgb.train(params,
                    lgb_train,
                    10000,
                    valid_sets=[lgb_train, lgb_val],
                    verbose_eval=200,
                    early_stopping_rounds=200)
    
    lgb_preds = clf.predict(df_feat_val.drop(['id', 'label', 'label_rate'], axis=1))

    valid_pred.loc[df_feat_val.index, 'lgb_prob'] = lgb_preds
    
    pickle.dump(clf, open(f'./result/lgb_fold{fold}.pkl','wb+'))

assert len(valid_pred[valid_pred.lgb_prob==-1]) == 0
# valid_pred = valid_pred[valid_pred.lgb_prob!=-1]

pickle.dump(valid_pred, open(f'./result/lgb_valid_pred.pkl','wb+'))
pickle.dump([t for t in list(df_feat.columns) if t not in ['id', 'label', 'label_rate']], open(f'./result/lgb_columns.pkl','wb+'))

inter_thresh = { 
    "Lead": 0.15,
    "Position": 0.15,
    "Evidence": 0.15,
    "Claim": 0.25,
    "Concluding Statement": 0.15,
    "Counterclaim": 0.25,
    "Rebuttal": 0.25,
}
def post_choice(df):
    rtn = []
    for k,group in tqdm(df.groupby(['id','class'])):
        group = group.sort_values('lgb_prob',ascending=False)

        preds_range = []
        for irow, row in group.iterrows():
            start = row.word_start
            end = row.word_end
            L1 = end-start+1
            flag = 0
            if L1 == 0:
                continue
            for pos_range in preds_range:
                L2 = pos_range[1] - pos_range[0] + 1
                intersection = (min(end, pos_range[1]) - max(start, pos_range[0]) + 1) / L1
                inter_t = inter_thresh[row['class']]
                if intersection>inter_t and (inter_t<=L1/L2<=1 or inter_t<=L2/L1<=1):
                    flag = 1
                    break

            if flag == 0:
                preds_range.append((start, end, row.lgb_prob))
                
                predictionstring = ' '.join(list(map(str, range(int(row.word_start), int(row.word_end)+1))))
                rtn.append((row.id, row['class'], predictionstring, row.lgb_prob))
    rtn = pd.DataFrame(rtn, columns=['id','class','predictionstring', 'lgb_prob'])
    return rtn

valid_pred_choice = post_choice(valid_pred)


proba_thresh = {
    "Lead": 0.45,
    "Position": 0.4,
    "Evidence": 0.45,
    "Claim": 0.35,
    "Concluding Statement": 0.5,
    "Counterclaim": 0.3,
    "Rebuttal": 0.3,
}


def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(" "))
    set_gt = set(row.predictionstring_gt.split(" "))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp_micro(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = (
        gt_df[["id", "discourse_type", "predictionstring"]]
        .reset_index(drop=True)
        .copy()
    )
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df["pred_id"] = pred_df.index
    gt_df["gt_id"] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(
        gt_df,
        left_on=["id", "class"],
        right_on=["id", "discourse_type"],
        how="outer",
        suffixes=("_pred", "_gt"),
    )
    joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
    joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")

    joined["overlaps"] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
    joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])

    joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
    joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)
    tp_pred_ids = (
        joined.query("potential_TP")
        .sort_values("max_overlap", ascending=False)
        .groupby(["id", "predictionstring_gt"])
        .first()["pred_id"]
        .values
    )

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    f1_score = TP / (TP + 0.5 * (FP + FN))
    precise_score = TP / (TP+FP)
    recall_score = TP / (TP+FN)
    
    return {'f1':f1_score, 'precise':precise_score, 'recall':recall_score}


def score_feedback_comp(pred_df, gt_df, return_class_scores=True):
    class_scores = {}
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    for discourse_type, gt_subset in gt_df.groupby("discourse_type"):
        pred_subset = (
            pred_df.loc[pred_df["class"] == discourse_type]
            .reset_index(drop=True)
            .copy()
        )
        class_scores[discourse_type] = score_feedback_comp_micro(pred_subset, gt_subset)
    f1 = np.mean([v['f1'] for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1


train_oof = train_df.copy()
res = {}
for k,v in proba_thresh.items():
    sub = valid_pred_choice[(valid_pred_choice.lgb_prob>v)&(valid_pred_choice['class']==k)]
    score_now = score_feedback_comp(sub, train_oof)[1][k]['f1']
    sub = valid_pred_choice[(valid_pred_choice.lgb_prob>v-0.05)&(valid_pred_choice['class']==k)]
    score_now1 = score_feedback_comp(sub, train_oof)[1][k]['f1']
    sub = valid_pred_choice[(valid_pred_choice.lgb_prob>v+0.05)&(valid_pred_choice['class']==k)]
    score_now2 = score_feedback_comp(sub, train_oof)[1][k]['f1']

    if max(score_now, score_now1, score_now2) == score_now:
        res[k] = (v, score_now)
        
    elif max(score_now, score_now1, score_now2) == score_now1:
        best_score = score_now1
        score_now3 = best_score
        i = 2
        while score_now3 >= best_score:
            sub = valid_pred_choice[(valid_pred_choice.lgb_prob>v-0.05*i)&(valid_pred_choice['class']==k)]
            score_now3 = score_feedback_comp(sub, train_oof)[1][k]['f1']
            best_score = max(best_score, score_now3)
            i += 1
            if v-0.05*i<=0:
                break
        res[k] = (v-0.05*(i-2), best_score)
        
    elif max(score_now, score_now1, score_now2) == score_now2:
        best_score = score_now2
        score_now3 = best_score
        i = 2
        while score_now3 >= best_score:
            sub = valid_pred_choice[(valid_pred_choice.lgb_prob>v+0.05*i)&(valid_pred_choice['class']==k)]
            score_now3 = score_feedback_comp(sub, train_oof)[1][k]['f1']
            best_score = max(best_score, score_now3)
            i += 1
            if v+0.05*i>=1:
                break
        res[k] = (v+0.05*(i-2), best_score)       


for k,v in res.items():
    logging.info(f"{k}:{v}")
logging.info(f"====== final score: {np.mean([v[1] for v in res.values()])} ======")

