import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from randk.utils import *
from sklearn.metrics import accuracy_score

def init_agg_grad(model):
    agg_grad = {}
    for name, param in model.named_parameters():
        agg_grad[name] = 0
    return agg_grad

def global_topk_sparsify(model, k_ratio=0.01):
    """
    Applies global top-k sparsification across all gradients in the model.
    Only the top-k elements (by absolute value) across all parameters are kept.
    """
    grads = []
    shapes = []
    for param in model.parameters():
        if param.grad is not None:
            g = param.grad.detach().view(-1)
            grads.append(g)
            shapes.append(param.grad.shape)

    # Flatten all grads into one vector
    flat_grads = torch.cat(grads)
    k = int(k_ratio * flat_grads.numel())

    if k == 0:
        return  # no gradient will be kept

    # Get top-k indices
    _, topk_indices = torch.topk(flat_grads.abs(), k, sorted=False)

    # Create sparse vector
    sparse_grads = torch.zeros_like(flat_grads)
    sparse_grads[topk_indices] = flat_grads[topk_indices]

    # Reshape and assign back to model parameters
    pointer = 0
    for param in model.parameters():
        if param.grad is not None:
            numel = param.grad.numel()
            param.grad.copy_(sparse_grads[pointer:pointer+numel].view_as(param.grad))
            pointer += numel
    return model

if __name__ == "__main__":
    args = get_args()

    training_data, test_data = get_data(args.dataset, args)
    train_dataloader = DataLoader(training_data, batch_size=args.train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
    # print(len(training_data))

    # Example usage with ResNet-18
    model = load_model(args)
    # print(model)

    # Apply sparsification after each optimizer step during training
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0
    patience = args.early_stop
    sparsities = []
    with tqdm(
            total=len(train_dataloader)*args.epochs, unit='batch'
                ) as pbar:
        for epoch in range(args.epochs):
            losses = []
            agg_grad = init_agg_grad(model)
            for i, (inputs, targets) in enumerate(train_dataloader):
                # print(inputs.shape)
                # print(targets)
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                losses.append(loss.item())
                # Apply sparsification to maintain sparsity
                total_param = 0
                zero_param = 0
                if args.k_sparse == "global_topk":
                    model = global_topk_sparsify(model, k_ratio=args.k_ratio)
                for name, param in model.named_parameters():
                    # # print(param.grad == 0)
                    # if "fc" in name:
                    #     # print(name)
                    if args.k_sparse == "layer_topk":
                        # print("a")
                        param.grad = layerwise_topk_sparsify(param.grad, k_ratio=args.k_ratio)
                    zero_param += torch.sum(param.grad == 0)
                    total_param += param.grad.numel()
                    agg_grad[name] += param.grad
                sparsities.append((zero_param/total_param).item())
                # print("model_size:", total_param)
                pbar.update(1)
                optimizer.zero_grad()
                if (i+1) % args.client_size == 0:
                    for name, param in model.named_parameters():
                        param.grad = agg_grad[name]/args.client_size
                    optimizer.step()
                    agg_grad = init_agg_grad(model)
                avg_loss = sum(losses)/len(losses)
                avg_sparse = sum(sparsities)/len(sparsities)
                pbar.set_postfix(loss=avg_loss, best_acc=best_acc, sparsity=avg_sparse)

            all_targets = []
            all_preds = []
            for inputs, targets in test_dataloader:
                all_targets.extend(targets.tolist())
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                with torch.no_grad():
                    output = model(inputs)
                _, predictions = torch.max(output, 1)
                all_preds.extend(predictions.tolist())
            accuracy = accuracy_score(all_targets, all_preds)

            if accuracy > best_acc:
                best_acc = accuracy
                patience = args.early_stop
                pbar.set_postfix(loss=avg_loss, best_acc=best_acc)
            else:
                patience -= 1
                if patience == 0:
                    finish = True
                    break