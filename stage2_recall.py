import pickle
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

df = pickle.load(open('./data/data_6model_offline712_online704_ensemble.pkl','rb'))

train_df = pd.read_csv('./data/train.csv')
IDS = train_df.id.unique()

dic_off_map = df[['id','offset_mapping']].set_index('id')['offset_mapping'].to_dict()
dic_txt = df[['id','text']].set_index('id')['text'].to_dict()

class CONFIG:
    def __init__(self):
        self.max_length = 4096
        
config = CONFIG()

id2label = {0:'Lead', 1:'Position', 2:'Evidence', 3:'Claim', 4:'Concluding Statement',
             5:'Counterclaim', 6:'Rebuttal', 7:'blank'}
label2id = {v:k for k,v in id2label.items()}

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

preds1_mean = {}
preds2_mean = {}
for irow,row in df.iterrows():
    t1, t2 = change_label(row.pred)
    preds1_mean[row.id] = t1
    preds2_mean[row.id] = t2

all_predictions = []

recall_thre = { 
    "Lead": 0.06,
    "Position": 0.05,
    "Evidence": 0.06,
    "Claim": 0.05,
    "Concluding Statement": 0.06,
    "Counterclaim": 0.02,
    "Rebuttal": 0.015,
}

# recall sample
for id in tqdm(preds1_mean):

    pred1_np = np.array(preds1_mean[id])
    pred2_np_all = np.array(preds2_mean[id])

    off_map = dic_off_map[id]
    off_map_len = len(off_map) if off_map[-1][1] != 0 else len(off_map)-1
    max_length = min(config.max_length, off_map_len)
    for class_num in range(7):
        thre = recall_thre[id2label[class_num]]
        pred2_np = pred2_np_all[:, class_num]
        
        i_start = 0
        while i_start < max_length:
            i = 0
            if pred1_np[i_start] > thre and pred2_np[i_start:i_start+10].max() > thre: #开头 两个阈值
                i = i_start + 1
                if i>=max_length: break
                while pred1_np[i] < (1-thre) and pred2_np[i:i+10].max() > thre: # 是否结束 两个阈值
                    cond = any([
                        i+1==max_length,
                        pred1_np[i] > thre,
                        i+1<max_length and pred2_np[i] < 0.7 and pred2_np[i] - pred2_np[i+1] > thre
                    ])
                    if i>i_start+1 and cond:
                        all_predictions.append((id, id2label[class_num], [i_start, i]))
                    i += 1
                    if i>=max_length: break
            
            if i != 0:
                if i == max_length:
                    i -=1

                all_predictions.append((id, id2label[class_num], [i_start, i]))
            i_start += 1

print(len(all_predictions))
valid_pred = pd.DataFrame(all_predictions, columns=['id', 'class', 'pos'])

predictionstring = []
for cache in tqdm(valid_pred.values):
    id = cache[0]
    pos = cache[2]
    off_map = dic_off_map[id]
    txt = dic_txt[id]
    txt_max = len(txt.split())
    
    start_word = len(txt[:off_map[pos[0]][0]].split())
    
    L = len(txt[off_map[pos[0]][0]:off_map[pos[1]][1]].split())
    end_word = min(txt_max, start_word+L) - 1
    
    predictionstring.append((start_word, end_word))
    
valid_pred['predictionstring'] = predictionstring


L_k = {
    "Evidence": 0.85,
    "Rebuttal": 0.6,
}

# select sample with high boundary threshold and choice 65% length with the highest probability of the current class as a new sample
def deal_predictionstring(df):
    new_predictionstring = []
    new_pos_list = []
    flag_list = []
    thre = 0.75
    for id, typ, pos, (start, end) in tqdm(df.values):
        flag = 0
        L = round(max(1, (pos[1]-pos[0]+1)*0.25))

        pos_left = max(0, pos[0]-L)
        pos_right = min(len(preds1_mean[id]), pos[1]+1+L)

        if start<10:
            left_thre = 2
        else:
            left_thre = max(preds1_mean[id][pos[0]], 1-preds2_mean[id][pos_left:pos[0],label2id[typ]].min())
        
        if pos[1] >= len(preds1_mean[id])-10:
            right_thre=2
        else:
            right_thre = max(preds1_mean[id][pos[1]+1:pos_right].max(), 1-preds2_mean[id][pos[1]+1:pos_right, label2id[typ]].min())
        
        if left_thre>thre and right_thre>thre:

            L = math.ceil((pos[1]-pos[0]+1)*L_k.get(typ, 0.65))

            tmp = {}
            for i in range(pos[0], pos[1]):
                if i+L>pos[1]:
                    break
                tmp[i] = np.sum(preds2_mean[id][i:i+L+1,label2id[typ]])
            if len(tmp)==0:
                new_pos = pos
            else:
                flag = min(left_thre, right_thre)

                new_start = max(tmp.keys(), key=lambda x:tmp[x])
                new_pos = (new_start,new_start+L)

        else:
            new_pos = pos

        off_map = dic_off_map[id]
        txt = dic_txt[id]
        txt_max = len(txt.split())

        start_word = len(txt[:off_map[new_pos[0]][0]].split())

        L = len(txt[off_map[new_pos[0]][0]:off_map[new_pos[1]][1]].split())
        end_word = min(txt_max, start_word+L) - 1

        new_predictionstring.append((start_word, end_word))
        new_pos_list.append(new_pos)
        flag_list.append(flag)
        
    df_new = df.copy()
    df_new['pos'] = new_pos_list
    df_new['predictionstring'] = new_predictionstring
    df_new['flag'] = flag_list
    
    df_new = pd.concat([df_new, df.loc[df_new[(df_new.flag>=thre) & (df_new.flag<0.95)].index]])
    df_new = df_new.reset_index(drop=True)
    df_new['flag'].fillna(0,inplace=True)
    
    return df_new


valid_pred = deal_predictionstring(valid_pred)
valid_oof = train_df.copy()
tmp = valid_oof.predictionstring.map(lambda x:x.split())
tmp1 = [(int(x[0]),int(x[-1])) for x in tmp]
valid_oof['predictionstring'] = tmp1

def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    try:
        start_pred, end_pred = row.predictionstring_pred
        start_gt, end_gt = row.predictionstring_gt
    except:
        return [0,0]

    # Length of each and intersection
    len_gt = end_gt - start_gt + 1
    len_pred = end_pred - start_pred + 1
    inter = min(end_pred, end_gt) - max(start_pred, start_gt) + 1
    overlap_1 = inter / (len_gt+1e-5)
    overlap_2 = inter / (len_pred+1e-5)
    return [overlap_1, overlap_2]


gt_df = (
    valid_oof[["id", "discourse_type", "predictionstring"]]
    .reset_index(drop=True)
    .copy()
)
pred_df = valid_pred[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
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
joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])

joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)
joined["min_overlap"] = joined[["overlap1", "overlap2"]].min(axis=1)

valid_pred['label'] = 0
valid_true_id = joined[joined.potential_TP==True]['pred_id']

valid_pred.loc[valid_true_id, 'label'] = 1

overlap = joined[['pred_id', 'min_overlap']]
overlap = overlap[~ overlap.pred_id.isna()]
overlap = overlap.groupby('pred_id')['min_overlap'].max().reset_index()

valid_pred = valid_pred.merge(overlap, left_index=True, right_on='pred_id', how='left')
valid_pred = valid_pred.drop('pred_id',axis=1)

pickle.dump(valid_pred, open('./data/recall_data.pkl','wb+'))