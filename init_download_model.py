import os
import copy
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModel, AutoConfig



# model = AutoModel.from_pretrained('t5-large') # 768

from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForMaskedLM
model_names = [
    'microsoft/deberta-large',
    'sshleifer/distilbart-xsum-12-6',
    'sshleifer/distilbart-cnn-12-6',
    'valhalla/bart-large-finetuned-squadv1',
    'valhalla/distilbart-mnli-12-9',
    'allenai/longformer-large-4096',
    'microsoft/deberta-v2-xlarge',
    'microsoft/deberta-v2-xxlarge',
    ]
for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)


    model_name = model_name.split('/')[0]
    import shutil
    from pathlib import Path
    import os
    import re
    cache_path = Path("./.cache/huggingface/transformers/")
    target_path = Path("./" + model_name)
    target_path.mkdir(exist_ok=True, parents=True)
    for filename in os.listdir(cache_path):
        if not filename.endswith(".json"):
            continue
        with open(cache_path / filename, "r") as f:
            content = f.read()
        if content.find(f"/{model_name}/") == -1:
            continue
        filename = filename[:-5]
        print(content)
        print(re.findall(f"/{model_name}/(.*?)\"", content))
        print()
        target_filename = re.findall(f"/{model_name}/(.*?)\"", content)[0]
        try:
            shutil.copyfile(
                cache_path / filename,
                target_path / target_filename
            )
        except:
            shutil.copyfile(
                cache_path / filename,
                target_path / target_filename.replace('resolve/main/','')
            )
print(os.listdir('~/'))