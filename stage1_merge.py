import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA_LAUNCH_BLOCKING=1
import pickle
import random
import numpy as np 
import pandas as pd 
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pdb
import torch
from torch import nn
from torch import cuda

# 线下评估测试
from utils.utils import Log, get_f1_score, after_deal
from utils.utils import get_feat, dataset, Collate, model_predict
from utils.utils import train, test
from utils.models import TextModel,TextModel9,FeedbackModel

import warnings
warnings.filterwarnings('ignore')


import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_path', type=str, default="log.txt")
    parser.add_argument('--data_path', type=str, default="feedback/")
    parser.add_argument('--text_path', type=str, default="feedback/train/")
    parser.add_argument('--cache_path', type=str, default="cache/")
    parser.add_argument('--model_name', type=str, default='roberta-large')
    parser.add_argument('--model_length', type=int, default=512)
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument("--fold", type=int,  default=1)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--valid_batch_size', type=int, default=1)
    parser.add_argument('--train_padding_side', type=str, default='random')
    parser.add_argument('--test_padding_side', type=str, default='right')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--min_lr', type=float, default=0.000001)
    parser.add_argument('--max_lr', type=float, default=0.00001)   
    parser.add_argument('--adv_lr', type=float, default=0.0000)  
    parser.add_argument('--adv_eps', type=float, default=0.001)  
    parser.add_argument('--max_grad_norm', type=float, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_feat', action='store_true')
    parser.add_argument('--key_string', type=str, default='')
        
    args = parser.parse_args()
    
    if 'longformer' in args.model_name:
        args.model_length=None
    elif 'bigbird' in args.model_name:
        args.model_length=None
    elif 'funnel-transformer' in args.model_name:
        args.model_length=None
    elif 'roberta' in args.model_name:
        args.model_length=512
        args.train_padding_side='random'
    elif 'albert-xxlarge-v2' in args.model_name:
        args.model_length=512
        args.train_padding_side='random'
    elif 'bert-large-NER' in args.model_name:
        args.model_length=512
        args.train_padding_side='random'
    elif 'electra' in args.model_name:
        args.model_length=512
        args.train_padding_side='random'
    elif 'gpt2' in args.model_name:
        args.model_length=1024
    elif 'distilbart' in args.model_name:
        args.model_length=1024
        args.train_padding_side='random'
    elif 'bart-large' in args.model_name:
        args.model_length=1024
        args.train_padding_side='random'
    elif 'deberta' in args.model_name:
        args.model_length=None
    else:
        args.model_length=None
    args.padding_dict = {'input_ids':0,'attention_mask':0,'labels':-100}
    args.epochs = 2 if args.debug else args.epochs
    args.key_string = args.model_name.split('/')[-1] + \
            '_v2_15class' + \
            f'_padding_{args.train_padding_side}' + \
            (f'_adv{args.adv_lr}' if args.adv_lr>0 else '') + \
            f'_fold_all' + \
            ('_debug' if args.debug else '')
    
    log = Log(f'log/{args.key_string}.log',time_key=False)
    log('args:{}'.format(str(args)))
    return args,log

# Function to seed everything
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

args,log = get_args()
os.listdir(args.cache_path)
path_list = [m for m in os.listdir(args.cache_path) if args.model_name in m and 'fold5' not in m]

seed_everything(args.seed)
discourse_type = ['Claim','Evidence', 'Position','Concluding Statement','Lead','Counterclaim','Rebuttal']
i_discourse_type = ['I-'+i for i in discourse_type]
b_discourse_type = ['B-'+i for i in discourse_type]
args.labels_to_ids = {k:v for v,k in enumerate(['O']+i_discourse_type+b_discourse_type)}
args.ids_to_labels = {k:v for v,k in args.labels_to_ids.items()}

test_df = pd.read_csv(args.data_path+'train_folds.csv')
log('test_df.shape:',test_df.shape,'\t')



def mapping_align_helper(offset_mapping1,offset_mapping2,preds1):
    preds2 = []
    idx1 = 0
    for idx2 in range(len(offset_mapping2)):
        pred2 = []
        mapping2 = offset_mapping2[idx2]
        try:
            if mapping2[0] == mapping2[1]:
                pred2.append(preds1[idx1])
                idx1 += 1
            else:
                while mapping2[0] >= offset_mapping1[idx1][1]:   # 赶超mapping
                    idx1 += 1
                if mapping2[0] <= offset_mapping1[idx1][0] and \
                    mapping2[1] >= offset_mapping1[idx1][1]:    # 完全包含
                    pred2.append(preds1[idx1])
                    idx1 += 1
                elif mapping2[0] < offset_mapping1[idx1][1] and \
                    mapping2[1] > offset_mapping1[idx1][0]:    # 有交集
                    pred2.append(preds1[idx1])
                    idx1 += 1
        except:
            pass
        if len(pred2) > 0:
            pred2 = np.mean(pred2,axis=0)
        else:
            pred2 = preds2[-1]
        preds2.append(pred2)
    return np.array(preds2)


from joblib import Parallel, delayed
def mapping_align(df1,df2):
    df2 = df1[['id']].merge(df2,on='id',how='left').reset_index(drop=True)
    num_jobs = 8
    params = zip(df2['offset_mapping'],df1['offset_mapping'],df2['pred'])
    new_pred = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(mapping_align_helper)(*param) for param in params
    )
    df2['pred'] = new_pred
    return df2


# 0.700026679940758
data_pred_right  = pickle.load(open('cache/data_pred_longformer-large-4096.pkl','+rb')).reset_index(drop=True)


data_pred_roberta   = pickle.load(open('cache/data_pred_roberta-large.pkl','+rb'))
data_pred_roberta = mapping_align(data_pred_right,data_pred_roberta)

data_pred_xsum = pickle.load(open('cache/data_pred_distilbart-xsum-12-6.pkl','+rb'))
data_pred_xsum = mapping_align(data_pred_right,data_pred_xsum)

data_pred_cnn = pickle.load(open('cache/data_pred_distilbart-cnn-12-6.pkl','+rb'))
data_pred_cnn = mapping_align(data_pred_right,data_pred_cnn)

data_pred_debertax = pickle.load(open('cache/data_pred_deberta-v2-xlarge.pkl','+rb'))
data_pred_debertax = mapping_align(data_pred_right,data_pred_debertax)

data_pred_deberta_v2_xxlarge = pickle.load(open('cache/data_pred_deberta-v2-xxlarge.pkl','+rb'))
data_pred_deberta_v2_xxlarge = mapping_align(data_pred_right,data_pred_deberta_v2_xxlarge)

data_pred_distilbart12_9 = pickle.load(open('cache/data_pred_distilbart-mnli-12-9.pkl','+rb'))
data_pred_distilbart12_9 = mapping_align(data_pred_right,data_pred_distilbart12_9)

data_pred_squadv1 = pickle.load(open('cache/data_pred_bart-large-finetuned-squadv1.pkl','+rb'))
data_pred_squadv1 = mapping_align(data_pred_right,data_pred_squadv1)




data_pred = data_pred_right.copy()
data_pred['pred'] = (
                      data_pred_right['pred']*1 + \
                      data_pred_roberta['pred']*1.5 + \
                      data_pred_xsum['pred']*0.5 + \
                      data_pred_cnn['pred']*0.5 + \
                      data_pred_debertax['pred']*0.5 + \
                      data_pred_deberta_v2_xxlarge['pred']*2.5 + \
                      data_pred_distilbart12_9['pred']*0.5 + \
                      data_pred_squadv1['pred']*0.5 + \
                      0
                     )
data_pred['pred'] = data_pred['pred'].apply(lambda x: x/x.sum(axis=1).reshape(-1,1))

segment_param = {
"Lead":                 {'min_proba':[0.47,0.41],'begin_proba':1.00,'min_sep':40,'min_length': 5},
"Position":             {'min_proba':[0.45,0.40],'begin_proba':0.90,'min_sep':21,'min_length': 3},
"Evidence":             {'min_proba':[0.50,0.40],'begin_proba':0.56,'min_sep': 2,'min_length':21},
"Claim":                {'min_proba':[0.40,0.30],'begin_proba':0.30,'min_sep':10,'min_length': 1},
"Concluding Statement": {'min_proba':[0.58,0.25],'begin_proba':0.93,'min_sep':50,'min_length': 5},
"Counterclaim":         {'min_proba':[0.45,0.25],'begin_proba':0.70,'min_sep':35,'min_length': 6},
"Rebuttal":             {'min_proba':[0.37,0.34],'begin_proba':0.70,'min_sep':45,'min_length': 5},
}

test_predictionstring = after_deal(data_pred, args.labels_to_ids, segment_param,log)
f1_score = get_f1_score(test_predictionstring,test_df,log,slient=False)

pickle.dump(data_pred,open('./data/data_6model_offline712_online704_ensemble.pkl','+wb'))
