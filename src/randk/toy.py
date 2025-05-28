import torch.nn as nn
import torch

m = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = m(input)
print(torch.sum(output==0)/output.numel())