
# 1
#################################################################################
#################################### load model #################################
#################################################################################
python init_download_model.py


# 2
#################################################################################
############################ split train.csv to 5 flod ##########################
#################################################################################
python init_split_data.py

# 3
#################################################################################
################################# cv5 + all_train_data ##########################
#################################################################################

################################# deberta-v2-xxlarge ##########################
for i in {0..5}  
do  
	 python -u ./stage1_train_eval.py \
	 --data_path "feedback/" \
	 --text_path "feedback/train/" \
	 --cache_path "cache/" \
	 --model_name "deberta-v2-xxlarge" \
	 --train_padding_side 'right' \
	 --test_padding_side 'right' \
	 --train_batch_size 4 \
	 --valid_batch_size 4 \
	 --adv_lr 0.0005 \
	 --adv_eps 0.001 \
	 --epochs 15 \
	 --max_lr 0.00001 \
	 --fold $i \
	 --max_length 1600 \
	 --seed 6
done 

################################# deberta-v2-xlarge ##########################
for i in {0..5}  
do  
	 python -u ./stage1_train_eval.py \
	 --data_path "feedback/" \
	 --text_path "feedback/train/" \
	 --cache_path "cache/" \
	 --model_name "deberta-v2-xlarge" \
	 --train_padding_side 'right' \
	 --test_padding_side 'right' \
	 --train_batch_size 4 \
	 --valid_batch_size 4 \
	 --adv_lr 0.0005 \
	 --adv_eps 0.001 \
	 --epochs 15 \
	 --max_lr 0.00001 \
	 --fold $i \
	 --max_length 1600 \
	 --seed 66
done 

################################# longformer-large-4096 ##########################
for i in {0..5}  
do  
	 python -u ./stage1_train_eval.py \
	 --data_path "feedback/" \
	 --text_path "feedback/train/" \
	 --cache_path "cache/" \
	 --model_name "longformer-large-4096" \
	 --train_padding_side 'random' \
	 --test_padding_side 'right' \
	 --train_batch_size 4 \
	 --valid_batch_size 4 \
	 --adv_lr 0.0005 \
	 --adv_eps 0.001 \
	 --epochs 15 \
	 --max_lr 0.00002 \
	 --fold $i \
	 --max_length 1600 \
	 --seed 666
done 

################################# distilbart-mnli-12-9 ##########################
for i in {0..5}  
do  
	 python -u ./stage1_train_eval.py \
	 --data_path "feedback/" \
	 --text_path "feedback/train/" \
	 --cache_path "cache/" \
	 --model_name "distilbart-mnli-12-9" \
	 --train_padding_side 'right' \
	 --test_padding_side 'right' \
	 --train_batch_size 4 \
	 --valid_batch_size 4 \
	 --adv_lr 0.0005 \
	 --adv_eps 0.001 \
	 --epochs 15 \
	 --max_lr 0.00002 \
	 --fold $i \
	 --max_length 1600 \
	 --seed 66666
done 

################################# bart-large-finetuned-squadv1 ##########################
for i in {0..5}  
do  
	 python -u ./stage1_train_eval.py \
	 --data_path "feedback/" \
	 --text_path "feedback/train/" \
	 --cache_path "cache/" \
	 --model_name "bart-large-finetuned-squadv1" \
	 --train_padding_side 'right' \
	 --test_padding_side 'right' \
	 --train_batch_size 4 \
	 --valid_batch_size 4 \
	 --adv_lr 0.0005 \
	 --adv_eps 0.001 \
	 --epochs 15 \
	 --max_lr 0.00002 \
	 --fold $i \
	 --max_length 1600 \
	 --seed 666666
done 

################################# roberta-large ##########################
for i in {0..5}  
do  
	 python -u ./stage1_train_eval.py \
	 --data_path "feedback/" \
	 --text_path "feedback/train/" \
	 --cache_path "cache/" \
	 --model_name "roberta-large" \
	 --train_padding_side 'random' \
	 --test_padding_side 'right' \
	 --train_batch_size 4 \
	 --valid_batch_size 4 \
	 --adv_lr 0.0005 \
	 --adv_eps 0.001 \
	 --epochs 15 \
	 --max_lr 0.00001 \
	 --fold $i \
	 --max_length 1600 \
	 --seed 6666666
done 

################################# distilbart-cnn-12-6 ##########################
for i in {0..5}  
do  
	 python -u ./stage1_train_eval.py \
	 --data_path "feedback/" \
	 --text_path "feedback/train/" \
	 --cache_path "cache/" \
	 --model_name "distilbart-cnn-12-6" \
	 --train_padding_side 'random' \
	 --test_padding_side 'right' \
	 --train_batch_size 4 \
	 --valid_batch_size 4 \
	 --adv_lr 0.0005 \
	 --adv_eps 0.001 \
	 --epochs 15 \
	 --max_lr 0.00001 \
	 --fold $i \
	 --max_length 1600 \
	 --seed 66666666
done 

################################# distilbart-xsum-12-6 ##########################
for i in {0..5}  
do  
	 python -u ./stage1_train_eval.py \
	 --data_path "feedback/" \
	 --text_path "feedback/train/" \
	 --cache_path "cache/" \
	 --model_name "distilbart-xsum-12-6" \
	 --train_padding_side 'random' \
	 --test_padding_side 'right' \
	 --train_batch_size 4 \
	 --valid_batch_size 4 \
	 --adv_lr 0.0005 \
	 --adv_eps 0.001 \
	 --epochs 15 \
	 --max_lr 0.000015 \
	 --fold $i \
	 --max_length 1600 \
	 --seed 66666666
done 


# 4
#################################################################################
################################# get_data_pred_cv5 ##########################
#################################################################################
python stage1_pred.py model_name "distilbart-xsum-12-6"
python stage1_pred.py model_name "distilbart-cnn-12-6"
python stage1_pred.py model_name "roberta-large"
python stage1_pred.py model_name "bart-large-finetuned-squadv1"
python stage1_pred.py model_name "distilbart-mnli-12-9"
python stage1_pred.py model_name "longformer-large-4096"
python stage1_pred.py model_name "deberta-v2-xlarge"
python stage1_pred.py model_name "deberta-v2-xxlarge"
python stage1_merge.py 

# get data_6model_offline712_online704_ensemble.pkl


# 5
#################################################################################
################################# stage two ##########################
#################################################################################
python stage2_recall_sample.py
python stage2_lsm_pca_train.py
python stage2_lgb_train.py