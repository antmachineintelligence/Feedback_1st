## Introduction

We divide our solution into two stages to reduce post-processing  
stage 1:  
15 class bert token prediction:  
&emsp;&emsp;	· huggingface download pretrain model  
&emsp;&emsp;	· split train.csv to 5 flod  
&emsp;&emsp;	· train 8 models  
&emsp;&emsp;&emsp;		deberta-v2-xxlarge  
&emsp;&emsp;&emsp;		deberta-v2-xlarge  
&emsp;&emsp;&emsp;		longformer-large-4096  
&emsp;&emsp;&emsp;		distilbart-mnli-12-9  
&emsp;&emsp;&emsp;		bart-large-finetuned-squadv1  
&emsp;&emsp;&emsp;		roberta-large  
&emsp;&emsp;&emsp;		distilbart-cnn-12-6   
&emsp;&emsp;&emsp;		distilbart-xsum-12-6  
&emsp;&emsp;	· Model weighted average ensemble to get the file:  
&emsp;&emsp;&emsp;      data_6model_offline712_online704_ensemble.pkl  


stage 2:  
lgb sentence prediction  
&emsp;&emsp;	· First recall as many candidate samples as possible by lowering the threshold. On the training set, we recall three million samples to achieve a mean of 95% of recalls.  
&emsp;&emsp;	· After getting the recall samples, we select sample with high boundary threshold and choice 65% length with the highest probability of the current class as a new sample.  
&emsp;&emsp;	· Finally, We made about 170 features for lightgbm training, and select samples as the final submission. 

### hardware
&emsp;&emsp;GPU: A100 * 4  
&emsp;&emsp;CPU: 60core +  
&emsp;&emsp;memory: 256G  

## Main files:
&emsp;&emsp; stage1_train_eval.py: train the Bert in stage1   
&emsp;&emsp; stage1_pred.py: generate the cv result   
&emsp;&emsp; stage1_merge.py: merge the cv result for next stage  
&emsp;&emsp; stage2_recall.py: generate the word segments as data for stage 2     
&emsp;&emsp; stage2_lstm_pca_train.py: train the lstm & pca model as features of lightgbm   
&emsp;&emsp; stage2_lgb_train.py: train the lightgbm model   
&emsp;&emsp; init.sh: the scripts to run all the pipeline   




