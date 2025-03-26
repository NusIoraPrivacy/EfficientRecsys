import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM
from peft import get_peft_model
from peft import LoraConfig, TaskType

# model_name = "bert-base-uncased"
# model_name = "bert-large-uncased"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "distilbert-base-uncased"
# model_name = "meta-llama/Llama-3.1-8B"
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"
model_name = 'distilbert-base-uncased'

# model = AutoModelForCausalLM.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
# print(model)
lora_config = LoraConfig(
    r=8,
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
    task_type=TaskType.CAUSAL_LM,
    # task_type=TaskType.SEQ_2_SEQ_LM,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)
# model.model.model.embed_tokens.weight.requires_grad=True
print(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def get_layer_sizes(model):
    layer_sizes = {}
    total_size = 0
    total_param = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_size = param.numel() * param.element_size()  # numel() returns the number of elements, element_size() returns the size in bytes of each element
            total_size += layer_size
            total_param += param.numel()
            layer_sizes[name] = (param.numel(), layer_size, param.dtype)

    return layer_sizes, total_size, total_param

layer_sizes, total_size, total_param = get_layer_sizes(model)

for name, size in layer_sizes.items():
    print(f"Layer: {name}; Number of parameters: {size[0]:,} ({size[2]}); Size: {size[1] / (1024 ** 2):.2f} MiB")

print(f"Total Number of parameters: {total_param:,}; Total Model Size: {total_size / (1024 ** 2):.2f} MiB")
print(f"Vocabulary size: {len(tokenizer):,}")