from datasets import Dataset
import copy
import torch
import argparse
import os

current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

task_to_keys = {
    "cola": ("sentence",),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence",),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def str2type(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, torch.dtype):
        return v
    if "float32" in v.lower():
        return torch.float32
    elif "float16" in v.lower():
        return torch.float16


def get_args():
    # python
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--dataset", type=str, default="mrpc")
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument('--train_batch_size', default=20, type=int)
    parser.add_argument('--test_batch_size', default=20, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--model', default="FacebookAI/roberta-large", type=str)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--top_k', action='store_true')
    parser.add_argument('--k_ratio', default=0.3, type=float)
    parser.add_argument('--client_size', default=1, type=int)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument("--early_stop", type=int, default=20, 
                        help = "number of rounds/patience for early stop")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--compress", type=str, default="none", choices=["none", "svd", "ternquant", "8intquant", "colr"])
    args = parser.parse_args()
    return args

class ClsDataset(Dataset):
    def __init__(self, inputs, tokenizer, max_words=100, pad=True, args=None):
        self.inputs = inputs
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad
        self.keys = task_to_keys[args.dataset]
        self.args = args

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        examples = []
        labels = []
        example_masks = []
        for i in index:
            sample = self.inputs[i]
            sentences = []
            for key in self.keys:
                sentences.append(sample[key])
            
            prompt = f" {self.tokenizer.bos_token} ".join(sentences)
            input_id = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.int64
            )

            if self.pad:
                padding = self.max_words - input_id.shape[0]
                if padding > 0:
                    input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
                elif padding < 0:
                    input_id = input_id[: self.max_words]

            att_mask = input_id.ge(0)
            input_id[~att_mask] = 0
            att_mask = att_mask.float()

            label = sample["label"]
            examples.append(input_id)
            labels.append(label)
            example_masks.append(att_mask)
        # print(examples)

        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask":example_masks,
        }

def global_topk_sparsify(model, emb_name, k_ratio=0.01):
    """
    Applies global top-k sparsification across all gradients in the model.
    Only the top-k elements (by absolute value) across all parameters are kept.
    """
    grads = []
    shapes = []
    for name, param in model.named_parameters():
        if param.grad is not None and name != emb_name:
            g = param.grad.detach().view(-1)
            grads.append(g)
            shapes.append(param.grad.shape)

    # Flatten all grads into one vector
    flat_grads = torch.cat(grads)
    k = int(k_ratio * flat_grads.numel())

    if k == 0:
        return  model# no gradient will be kept

    # Get top-k indices
    _, topk_indices = torch.topk(flat_grads.abs(), k, sorted=False)

    # Create sparse vector
    sparse_grads = torch.zeros_like(flat_grads)
    sparse_grads[topk_indices] = flat_grads[topk_indices]

    # Reshape and assign back to model parameters
    pointer = 0
    for name, param in model.named_parameters():
        if param.grad is not None and name != emb_name:
            numel = param.grad.numel()
            param.grad.copy_(sparse_grads[pointer:pointer+numel].view_as(param.grad))
            pointer += numel
    return model