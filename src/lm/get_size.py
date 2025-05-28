import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
from peft import get_peft_model, LoraConfig, TaskType
from lm.test_utility import model_to_emb

# model_name = "bert-base-uncased"
# model_name = "bert-large-uncased"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "distilbert-base-uncased"
# model_name = "meta-llama/Llama-3.1-8B"
# model_name = "Qwen/Qwen2.5-1.5B"
# model_name = "FacebookAI/roberta-large"
model_name = "meta-llama/Llama-3.3-70B-Instruct"

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, torch_dtype=torch.float16, device_map = 'auto') # , device_map = 'auto'
# print(model)
peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        # target_modules=["q_lin", "v_lin"]
    )
model = get_peft_model(model, peft_config)
# print(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
for name, param in model.named_parameters():
    if "emb" in name:
        print(name, param.shape)
emb_name = model_to_emb[model_name]

def get_layer_sizes(model):
    layer_sizes = {}
    total_size = 0
    total_param = 0
    emb_size = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_size = param.numel() * param.element_size()  # numel() returns the number of elements, element_size() returns the size in bytes of each element
            total_size += layer_size
            total_param += param.numel()
            layer_sizes[name] = (param.numel(), layer_size, param.dtype)
        elif name == emb_name:
            emb_size += param.numel()
        # if name == "base_model.model.classifier.modules_to_save.default.dense.weight":
        #     print(param.shape)

    return layer_sizes, total_size, total_param, emb_size

layer_sizes, total_size, total_param, emb_size = get_layer_sizes(model)

for name, size in layer_sizes.items():
    print(f"Layer: {name}; Number of parameters: {size[0]:,} ({size[2]}); Size: {size[1] / (1024 ** 2):.2f} MiB")

print(f"Total Number of parameters: {total_param:,}; Total Model Size: {total_size / (1024 ** 2):.2f} MiB")
print(f"Vocabulary size: {len(tokenizer):,}")
print(f"Word embedding size: {emb_size}")