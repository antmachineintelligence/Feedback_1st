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
    parser.add_argument('--model_name', type=str, default='')
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

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if 'deberta-v' in args.model_name or 'bigbird' in args.model_name:
    tokenizer.add_tokens('\n')
    
test_feat = get_feat(test_df,tokenizer,args,'test_feat'+args.key_string)
test_feat = test_feat.merge(test_df[['id','kfold']].drop_duplicates(),on='id',how='left')

log("test_feat.shap: {}".format(test_feat.shape),'\t')

test_params = {'batch_size': args.valid_batch_size,'shuffle': False, 'num_workers': 2,'pin_memory':True,
                   'collate_fn':Collate(padding_side='right',padding_dict=args.padding_dict)
                    }
# CREATE MODEL
model = TextModel(args.model_name, num_labels=len(args.labels_to_ids))
model.model.resize_token_embeddings(len(tokenizer))
# model = TextModel9(args.model_name, num_labels=9)
model = torch.nn.DataParallel(model)
model.to(args.device)
test_preds = []
for i,model_path in enumerate(path_list):

    test_feat_sub = test_feat[test_feat['kfold']==i%5].reset_index(drop=True)
    test_loader = DataLoader(dataset(test_feat_sub), **test_params)
    model.load_state_dict(torch.load( model_path))
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if args.test_padding_side in ['right','left']:
                loss, te_logits = model_predict(model, batch, model_length=args.model_length,max_length=args.max_length,
                                                padding_dict=args.padding_dict,padding_side=args.test_padding_side)
            elif args.test_padding_side == 'double':
                loss1, te_logits1 = model_predict(model, batch, model_length=args.model_length,max_length=args.max_length,
                                                padding_dict=args.padding_dict,padding_side='right')
                loss2, te_logits2 = model_predict(model, batch, model_length=args.model_length,max_length=args.max_length,
                                                padding_dict=args.padding_dict,padding_side='left')
                te_logits = [(l1+l2)/2 for l1,l2 in zip(te_logits1,te_logits2)]
            preds.extend(te_logits)
    if i < 5:
        test_feat_sub['pred'] = preds
        test_preds.append(test_feat_sub)
    else:
        test_preds[i%5]['pred'] = [p1+p2 for p1,p2 in zip(preds,test_preds[i%5]['pred'])]
test_preds = pd.concat(test_preds,axis=0)
test_preds['pred'] = test_preds['pred'].apply(lambda x:x.astype('float32'))

segment_param = {
"Lead":                 {'min_proba':[0.47,0.41],'begin_proba':1.00,'min_sep':40,'min_length': 5},
"Position":             {'min_proba':[0.45,0.40],'begin_proba':0.90,'min_sep':21,'min_length': 3},
"Evidence":             {'min_proba':[0.50,0.40],'begin_proba':0.56,'min_sep': 2,'min_length':21},
"Claim":                {'min_proba':[0.40,0.30],'begin_proba':0.30,'min_sep':10,'min_length': 1},
"Concluding Statement": {'min_proba':[0.58,0.25],'begin_proba':0.93,'min_sep':50,'min_length': 5},
"Counterclaim":         {'min_proba':[0.45,0.25],'begin_proba':0.70,'min_sep':35,'min_length': 6},
"Rebuttal":             {'min_proba':[0.37,0.34],'begin_proba':0.70,'min_sep':45,'min_length': 5},
}

# fold = 0
# test_predictionstring = after_deal(test_preds[test_preds.kfold==fold], args.labels_to_ids, segment_param,log)
# f1_score = get_f1_score(test_predictionstring,test_df[test_df.kfold==fold],log,slient=False)
test_predictionstring = after_deal(test_preds, args.labels_to_ids, segment_param,log)
f1_score = get_f1_score(test_predictionstring,test_df,log,slient=False)


pickle.dump(test_preds,open(f'cache/data_pred_{args.model_name}.pkl','+wb'))

