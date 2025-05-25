import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from lm.utils import task_to_keys
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType

# model_name = "bert-base-uncased"
# model_name = "bert-large-uncased"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "meta-llama/Llama-3.1-8B"
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"
# model_name = 'distilbert-base-uncased'
# model_name = "FacebookAI/roberta-large"
# model_name = "Qwen/Qwen2.5-1.5B"
model_name = "meta-llama/Llama-3.3-70B-Instruct"
data_name = "sst2"
keys = task_to_keys[data_name]

# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print("Model loaded")

dataset = load_dataset("nyu-mll/glue", data_name, split="train")
print(f"Data size:", len(dataset))
batch_size = int(len(dataset)/100)
print(f"Batch size:", batch_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("data loaded")

unq_cnts = []
for batch in tqdm(dataloader):
    token_inputs = []
    for key in keys:
        inputs = tokenizer(batch[key], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        token_inputs.append(inputs['input_ids'])
    token_inputs = torch.concatenate(token_inputs)
    unq_cnts.append(len(torch.unique(token_inputs)))
unq_cnt = sum(unq_cnts) / len(unq_cnts)
prop = unq_cnt/len(tokenizer) * 100
print(f"Vocabulary size: {len(tokenizer):,}; number of tokens: {unq_cnt:,}; proportion: {prop:.2f} %")