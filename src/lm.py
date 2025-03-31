import torch
from transformers import AutoModel

model_name = "meta-llama/Meta-Llama-3-8B"
model = AutoModel.from_pretrained(model_name)

def get_layer_sizes(model):
    layer_sizes = {}
    total_size = 0
    total_param = 0

    for name, param in model.named_parameters():
        layer_size = param.numel() * param.element_size()  # numel() returns the number of elements, element_size() returns the size in bytes of each element
        total_size += layer_size
        total_param += param.numel()
        layer_sizes[name] = (param.numel(), layer_size, param.dtype)

    return layer_sizes, total_size, total_param

layer_sizes, total_size, total_param = get_layer_sizes(model)

for name, size in layer_sizes.items():
    print(f"Layer: {name}; Number of parameters: {size[0]:,} ({size[2]}); Size: {size[1] / (1024 ** 2):.2f} MiB")

print(f"Total Number of parameters: {total_param:,}; Total Model Size: {total_size / (1024 ** 2):.2f} MiB")