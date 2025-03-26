import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import numpy as np

# model_name = "bert-base-uncased"
# model_name = "bert-large-uncased"
# model_name = "meta-llama/Meta-Llama-3-8B"
model_name = "meta-llama/Llama-3.1-8B"
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"
# model_name = 'distilbert-base-uncased'
batch_size = 100

model = AutoModel.from_pretrained(model_name, device_map = 'cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded")

dataset = load_dataset("tweet_eval", "sentiment", split="train")
print("data loaded")
unq_cnt = 0
for i in range(5):
    print(f"Round {i}")
    subset_indices = np.random.choice(len(dataset), min(batch_size, len(dataset)), replace=False)
    sample_dataset = dataset.select(subset_indices)
    inputs = tokenizer(sample_dataset['text'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    token_input = inputs['input_ids']
    unq_cnt = max(unq_cnt, len(torch.unique(token_input)))
prop = unq_cnt/len(tokenizer) * 100
print(f"Number of tokens: {unq_cnt:,}; proportion: {prop:.2f} %")