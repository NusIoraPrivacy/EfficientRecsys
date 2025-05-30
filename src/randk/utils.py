import torch
import argparse
import os
from torchvision import transforms, datasets, models
import torch.nn as nn
from randk.models import ResNet18, ResNet18Reduce
from datasets import load_dataset

current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

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
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--k_sparse", type=str, default=["none", "global_topk", "layer_topk"][0])
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--model', default=["ResNet18", "ResNet18Reduce"][0], type=str)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--client_size', default=100, type=int)
    parser.add_argument('--k_ratio', default=0.3, type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument("--early_stop", type=int, default=20, 
                        help = "number of rounds/patience for early stop")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--compress", type=str, default="svd", choices=["none", "svd", "ternquant", "8intquant", "colr"])
    args = parser.parse_args()
    return args

def random_k_sparsify(tensor, sparsity):
    """
    Randomly zero out a fraction of the tensor's elements to achieve the desired sparsity.
    """
    if sparsity <= 0.0:
        return tensor
    elif sparsity >= 1.0:
        return torch.zeros_like(tensor)
    
    tensor_flat = tensor.view(-1)
    num_elements = tensor_flat.numel()
    k = int((1 - sparsity) * num_elements)
    
    # Generate a mask with k ones and the rest zeros
    mask = torch.zeros(num_elements, device=tensor.device)
    indices = torch.randperm(num_elements, device=tensor.device)[:k]
    mask[indices] = 1
    mask = mask.view(tensor.size())
    
    return tensor * mask

def layerwise_topk_sparsify(grad, k_ratio=0.01):
    """
    Sparsifies the gradient tensor by keeping only top-k% values by magnitude.
    """
    if grad is None:
        return grad
    grad_flat = grad.view(-1)
    k = int(k_ratio * grad_flat.numel())
    if k == 0:
        return torch.zeros_like(grad)
    
    # Get top-k indices
    _, topk_indices = torch.topk(torch.abs(grad_flat), k, sorted=False)
    
    # Create sparse gradient
    sparse_grad = torch.zeros_like(grad_flat)
    sparse_grad[topk_indices] = grad_flat[topk_indices]
    
    return sparse_grad.view_as(grad)

def get_data(data_name, args):
    if data_name == "fashionmnist":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        training_data = datasets.FashionMNIST(
            root=f"{args.root_path}/data",
            train=True,
            download=True,
            transform=transform
        )

        test_data = datasets.FashionMNIST(
            root=f"{args.root_path}/data",
            train=False,
            download=True,
            transform=transform
        )
        return training_data, test_data

    elif data_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        training_data = datasets.MNIST(
            root=f"{args.root_path}/data",
            train=True,
            download=True,
            transform=transform
        )

        test_data = datasets.MNIST(
            root=f"{args.root_path}/data",
            train=False,
            download=True,
            transform=transform
        )
        return training_data, test_data
    
    elif data_name == "cifar10":
        stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        transform_train = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.Normalize(*stats),
                                ])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(*stats)])

        training_data = datasets.CIFAR10(root=f"{args.root_path}/data", train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR10(root=f"{args.root_path}/data", train=False, download=True, transform=transform_test)
        return training_data, test_data
    
    elif data_name == "cifar100":
        stats = ((0.50707515, 0.48654887, 0.44091784), (0.26733428, 0.256438462, 0.2761504713))

        transform_train = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.Normalize(*stats)])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(*stats)])

        training_data = datasets.CIFAR100(root=f"{args.root_path}/data", train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR100(root=f"{args.root_path}/data", train=False, download=True, transform=transform_test)
        return training_data, test_data
    
    elif data_name == "miniimagenet":
        training_data = load_dataset("timm/mini-imagenet", split="train")
        test_data = load_dataset("timm/mini-imagenet", split="test")
                # Define transformations
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # Apply transformations and create DataLoaders
        def transform_examples(example):
            example['image'] = transform(example['image'])
            return example
        training_data = training_data.map(transform_examples)
        test_data = test_data.map(transform_examples)
        return training_data, test_data

def load_model(args):
    if args.dataset in ("cifar100", "miniimagenet"):
        num_classes = 100
    else:
        num_classes = 10
    # model = models.resnet18(pretrained=False, num_classes=num_classes)
    model = eval(args.model)(dropout=args.dropout, num_classes=num_classes)
    if args.dataset in ("fashionmnist", "mnist"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model = model.to(args.device)
    return model