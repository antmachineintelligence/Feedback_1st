import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from utils.utils import get_feat, dataset, Collate
from utils.utils import train, test
from utils.models import TextModel,TextModel9,FeedbackModel

import warnings
warnings.filterwarnings('ignore')
torch.set_warn_always(False)

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_path', type=str, default="log.txt")
    parser.add_argument('--data_path', type=str, default="feedback/")
    parser.add_argument('--text_path', type=str, default="feedback/train/")
    parser.add_argument('--cache_path', type=str, default="cache/")
    parser.add_argument('--model_name', type=str, default='roberta-base/')
    parser.add_argument('--model_length', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument("--fold", type=int,  default=1)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--valid_batch_size', type=int, default=4)
    parser.add_argument('--train_padding_side', type=str, default='random')
    parser.add_argument('--test_padding_side', type=str, default='right')
    parser.add_argument('--epochs', type=int, default=10)
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
            '_aistudio_15class' + \
            f'_padding_{args.train_padding_side}' + \
            (f'_adv{args.adv_lr}' if args.adv_lr>0 else '') + \
            f'_max_lr_{args.max_lr}' + \
            f'_max_length_{args.max_length}' + \
            f'_fold{args.fold}' + \
            args.key_string + \
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


if __name__ == "__main__":
    
    args,log = get_args()
    seed_everything(args.seed)
    
    discourse_type = ['Claim','Evidence', 'Position','Concluding Statement','Lead','Counterclaim','Rebuttal']
    i_discourse_type = ['I-'+i for i in discourse_type]
    b_discourse_type = ['B-'+i for i in discourse_type]
    args.labels_to_ids = {k:v for v,k in enumerate(['O']+i_discourse_type+b_discourse_type)}
    args.ids_to_labels = {k:v for v,k in args.labels_to_ids.items()}
    
    df = pd.read_csv(os.path.join(args.data_path, "train_folds.csv"))
    
    train_df = df[df["kfold"] != args.fold].reset_index(drop=True)
    test_df = df[df["kfold"] == args.fold].reset_index(drop=True)
    if len(test_df) == 0:
        sample_id = train_df['id'].drop_duplicates().sample(frac=0.2).values
        test_df = train_df[train_df['id'].isin(sample_id)].reset_index(drop=True)

    if args.debug:
        sample_id = train_df['id'].drop_duplicates().sample(frac=0.01).values
        train_df = train_df[train_df['id'].isin(sample_id)].reset_index(drop=True)
        sample_id = test_df['id'].drop_duplicates().sample(frac=0.01).values
        test_df = test_df[test_df['id'].isin(sample_id)].reset_index(drop=True)
    log('train_df.shape:',train_df.shape,'\t','test_df.shape:',test_df.shape)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # if 'deberta-v' in args.model_name or 'bigbird' in args.model_name:
    #     tokenizer.add_tokens('\n')

    train_feat = get_feat(train_df,tokenizer,args,'train_feat'+args.key_string)
    test_feat = get_feat(test_df,tokenizer,args,'test_feat'+args.key_string)

    log("train_feat.shap: {}".format(train_feat.shape),'\t',"test_feat.shape: {}".format(test_feat.shape))

    train_params = {'batch_size': args.train_batch_size,'shuffle': True, 'num_workers': 2, 'pin_memory':True,
                    'collate_fn':Collate(args.model_length,args.max_length,args.train_padding_side,args.padding_dict)
                    }
    test_params = {'batch_size': args.valid_batch_size,'shuffle': False, 'num_workers': 2,'pin_memory':True,
                   'collate_fn':Collate(padding_side='right',padding_dict=args.padding_dict)
                    }
    train_loader = DataLoader(dataset(train_feat), **train_params)
    test_loader = DataLoader(dataset(test_feat), **test_params)
    args.num_train_steps = len(train_feat) * args.epochs / args.train_batch_size

    # CREATE MODEL
    model = TextModel(args.model_name, num_labels=len(args.labels_to_ids))
    # model.model.resize_token_embeddings(len(tokenizer))
    # model = TextModel9(args.model_name, num_labels=9)
    model = torch.nn.DataParallel(model)
    model.to(args.device)

    model_path = f'{args.cache_path + args.key_string}.pt'

    if args.load_model and os.path.exists(model_path):
        model.load_state_dict(torch.load( model_path))
        log(f'Model loaded from {model_path}')
        te_loss, te_accuracy,f1_score, test_pred = test(model,test_loader,test_feat,test_df,args,log)
    else:
        model,test_pred = train(model,train_loader,test_loader,test_feat,test_df,args,model_path,log)

        