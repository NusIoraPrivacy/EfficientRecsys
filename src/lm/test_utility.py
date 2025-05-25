from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    get_scheduler
)
from lm.utils import ClsDataset, get_args
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

def get_model_tokenizer_cls(model_name, num_labels, args):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if model_name in ('gpt2', 'llama'):
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(args.device)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.pad_token
        tokenizer.eos_token_id = tokenizer.pad_token_id
    return base_model, tokenizer

model_to_emb = {
    "FacebookAI/roberta-large": "base_model.model.roberta.embeddings.word_embeddings.weight",
    "meta-llama/Llama-3.1-8B": "base_model.model.model.embed_tokens.weight",
    "distilbert-base-uncased": "base_model.model.distilbert.embeddings.word_embeddings.weight",
    "Qwen/Qwen2.5-1.5B": "base_model.model.model.embed_tokens.weight",
}

model_to_target = {
    "FacebookAI/roberta-large": ["word_embeddings", "query", "value"]
}

if __name__ == "__main__":
    args = get_args()
    model, tokenizer = get_model_tokenizer_cls(args.model, 2, args)
    # print(model)
    if args.compress == "colr":
        targets = model_to_target[args.model]
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=targets
        )
        # print(peft_config)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    emb_name = model_to_emb[args.model]
    if args.compress != "colr":
        for name, param in model.named_parameters():
            if name == emb_name:
                param.requires_grad = True
            # if "embedding" in name:
            #     print(name)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    train_data = load_dataset("nyu-mll/glue", args.dataset, split="train")
    train_dataset = ClsDataset(train_data, tokenizer, args=args)
    test_data = load_dataset("nyu-mll/glue", args.dataset, split="validation")
    test_dataset = ClsDataset(test_data, tokenizer, args=args)
    
    train_dataloader = DataLoader(
            train_dataset, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            shuffle=True,
            batch_size=args.train_batch_size,
        )
    
    test_dataloader = DataLoader(
            test_dataset, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            batch_size=args.test_batch_size,
        )
    
    optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=0.0,
            )
    scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=len(train_dataloader)*args.epochs,
                )
    
    best_acc = 0
    best_auc = 0
    with tqdm(
        total=len(train_dataloader)*args.epochs, unit='batch'
            ) as pbar:
        for epoch in tqdm(range(args.epochs)):
            losses = []
            for batch in train_dataloader:
                for key in batch:
                    batch[key] = batch[key].to(args.device)
                output = model(**batch)
                loss = output.loss

                loss.backward()
                if args.compress == "svd":
                    for name, param in model.named_parameters():
                        if name == emb_name:
                            # print(name)
                            matrix_cpu = param.grad.cpu()
                            U_cpu, S_cpu, V_cpu = torch.svd(matrix_cpu)
                            U_cpu, S_cpu, V_cpu = U_cpu[:, :args.rank], S_cpu[:args.rank], V_cpu[:, :args.rank]
                            U, S, V = U_cpu.to(args.device), S_cpu.to(args.device), V_cpu.to(args.device)
                            param.grad = torch.mm(torch.mm(U, torch.diag(S)), V.t())
                
                elif args.compress == "ternquant":
                    for name, param in model.named_parameters():
                        if name == emb_name:
                            # print(name)
                            max_grad = torch.abs(param.grad).max().item()
                            probs = torch.abs(param.grad) / max_grad
                            rand_values = torch.rand(param.grad.shape, device=probs.device)
                            binary_vec = (rand_values >= probs).float()
                            ternary_grads = binary_vec * torch.sign(param.grad) * max_grad
                            param.grad = ternary_grads

                elif args.compress == "8intquant":
                    for name, param in model.named_parameters():
                        if name == emb_name:
                            quant_grad = torch.quantize_per_tensor(param.grad, 0.00001, 0, torch.qint8) 
                            quant_grad = torch.dequantize(quant_grad)
                            param.grad = quant_grad

                losses.append(loss.item())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)
                avg_loss = sum(losses)/len(losses)
                pbar.set_postfix(loss=avg_loss, best_acc=best_acc, best_auc=best_auc)
                # break
        
            all_labels = []
            all_preds = []
            for batch in test_dataloader:
                for key in batch:
                    batch[key] = batch[key].to(args.device)
                # print(batch)
                labels = batch["labels"]
                with torch.no_grad():
                    output = model(**batch)
                logits = output.logits
                preds = logits.argmax(dim=1)
                all_labels.extend(labels.tolist())
                all_preds.extend(preds.tolist())
            accuracy = accuracy_score(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_preds)
            best_acc = max(best_acc, accuracy)
            best_auc = max(best_auc, auc)