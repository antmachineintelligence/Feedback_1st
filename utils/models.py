import copy
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModel, AutoConfig


class TextModel(nn.Module):
    def __init__(self,model_name=None,num_labels=1):
        super(TextModel,self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name) # 768
        self.drop_out = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size,num_labels)
        
        if 'deberta-v2-xxlarge' in model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:24].requires_grad_(False) # 冻结24/48
        if 'deberta-v2-xlarge' in model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:12].requires_grad_(False) # 冻结12/24
        if 'funnel-transformer-xlarge' in model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.blocks[:1].requires_grad_(False) # 冻结1/3

    def forward(self, input_ids, attention_mask, labels=None):
        if 'gpt' in self.model.name_or_path:
            emb = self.model(input_ids)[0]
        else:
            emb = self.model(input_ids,attention_mask)[0]

        preds1 = self.output(self.dropout1(emb))
        preds2 = self.output(self.dropout2(emb))
        preds3 = self.output(self.dropout3(emb))
        preds4 = self.output(self.dropout4(emb))
        preds5 = self.output(self.dropout5(emb))
        preds = (preds1 + preds2 + preds3 + preds4 + preds5) / 5
        
        logits = torch.softmax(preds, dim=-1)
        if labels is not None:
            loss = self.get_loss(preds,labels,attention_mask)
            return loss,logits
        else:
            return logits
    
    def get_loss(self, outputs, targets, attention_mask):
        loss_fct = nn.CrossEntropyLoss()

        active_loss = attention_mask.reshape(-1) == 1
        active_logits = outputs.reshape(-1, outputs.shape[-1])
        true_labels = targets.reshape(-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = loss_fct(active_logits, true_labels)

        return loss   


class TextModel9(nn.Module):
    def __init__(self,model_name=None,num_labels=1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name) # 768
        self.drop_out = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size,num_labels)
        
        if 'deberta-v2-xxlarge' in model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:24].requires_grad_(False) # 冻结24/48
        if 'deberta-v2-xlarge' in model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:12].requires_grad_(False) # 冻结12/24

    def forward(self, input_ids, attention_mask, labels=None):
        if 'gpt' in self.model.name_or_path:
            emb = self.model(input_ids)[0]
        else:
            emb = self.model(input_ids,attention_mask)[0]

        preds1 = self.output(self.dropout1(emb))
        preds2 = self.output(self.dropout2(emb))
        preds3 = self.output(self.dropout3(emb))
        preds4 = self.output(self.dropout4(emb))
        preds5 = self.output(self.dropout5(emb))
        preds = (preds1 + preds2 + preds3 + preds4 + preds5) / 5
        
        logits1 = torch.softmax(preds[:,:,:8], dim=-1)
        logits2 = torch.sigmoid(preds[:,:,-1:])
        logits = torch.cat([logits1,logits2],dim=-1)
        
        if labels is not None:
            loss = self.get_loss(preds,labels,attention_mask)
            return loss,logits
        else:
            return logits

    def get_loss(self, outputs, targets, attention_mask):
        loss_fct = nn.CrossEntropyLoss()
        loss_bce = nn.BCEWithLogitsLoss()
        
        active = attention_mask.reshape(-1) == 1
        active_logits = outputs.reshape(-1, outputs.shape[-1])
        true_labels = targets.reshape(-1)
        idxs = np.where(active.cpu().numpy() == 1)[0]
        
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)
        calss_targets = (true_labels - (true_labels>=8)*8).long()
        head_targets = (true_labels>=8).float()

        loss1 = loss_fct(active_logits[:,:8], calss_targets)
        loss2 = loss_bce(active_logits[:,8],  head_targets)

        loss = loss1+loss2
        return loss     

    
class TextModel2(nn.Module):
    def __init__(self,model_name=None,num_labels=1):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name) # 768
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(768,num_labels)
        self.loss_function = CrossEntropyLoss(reduction='mean')
        
    def forward(self, input_ids, attention_mask, labels=None):
        emb = self.model(input_ids,attention_mask)[0]

        preds1 = self.output(self.dropout1(emb))
        preds2 = self.output(self.dropout2(emb))
        preds3 = self.output(self.dropout3(emb))
        preds4 = self.output(self.dropout4(emb))
        preds5 = self.output(self.dropout5(emb))
        preds = (preds1 + preds2 + preds3 + preds4 + preds5) / 5
        
        logits = torch.softmax(preds, dim=-1)
        if labels is not None:
            loss = self.get_loss(preds,labels)
            return loss,logits
        else:
            return logits
    
    def get_loss(self, preds, labels):   
        batch_size,lenght,num_labels = preds.shape
        preds = preds.view(-1,num_labels)
        labels = labels.view(-1)
        loss = self.loss_function(preds,labels)
        return loss
    
import numpy as np
class FeedbackModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(model_name)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": self.num_labels,
            }
        )
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size, self.num_labels)

    def loss(self, outputs, targets, attention_mask):
        loss_fct = nn.CrossEntropyLoss()

        active_loss = attention_mask.view(-1) == 1
        active_logits = outputs.view(-1, self.num_labels)
        true_labels = targets.view(-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = loss_fct(active_logits, true_labels)
        return loss

    def forward(self, input_ids, attention_mask, labels=None):

        transformer_out = self.transformer(input_ids, attention_mask)
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        logits = torch.softmax(logits, dim=-1)

        if labels is not None:
            loss1 = self.loss(logits1, labels, attention_mask=attention_mask)
            loss2 = self.loss(logits2, labels, attention_mask=attention_mask)
            loss3 = self.loss(logits3, labels, attention_mask=attention_mask)
            loss4 = self.loss(logits4, labels, attention_mask=attention_mask)
            loss5 = self.loss(logits5, labels, attention_mask=attention_mask)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            return loss, logits
        return logits
