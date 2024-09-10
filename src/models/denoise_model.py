import torch
from torch import nn

class MF_d(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, args):
        super(MF_d, self).__init__()
        self.embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=4)
        self.layer1 = nn.Sequential(
            nn.Linear(emb_dim, 4),
            # nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(emb_dim, 4),
            # nn.ReLU(),
        )
        self.p = nn.Parameter(torch.tensor([1.0]))
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.layer1.apply(self.init_layer)
        self.layer2.apply(self.init_layer)
    
    def init_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.zeros_(m.bias)
    
    def forward(self, ratings, item_ids, noise, init_user_emb):
        item_embs = self.embedding_item(item_ids) # (batch size, item size, reduce dim)
        noise = self.layer1(noise) # (batch size, reduce dim)
        user_emb = self.layer2(init_user_emb) # (batch size, reduce dim)
        noise = torch.einsum("ijk, ik->ij", item_embs, noise) # (batch size, item size)
        user_emb = torch.einsum("ijk, ik->ij", item_embs, user_emb) # (batch size, item size)
        denoise_output = self.p * ratings + noise + user_emb # (batch size, item size)
        return denoise_output

    def get_loss(self, true_preds, denoise_preds, mask=None):
        if mask is not None:
            loss = ((true_preds - denoise_preds) ** 2) * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = torch.mean((true_preds - denoise_preds) ** 2)
        reg_loss = self.embedding_item.weight.norm(2).pow(2)/self.num_items
        loss += reg_loss
        return loss

class NCF_d(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, args):
        super(NCF_d, self).__init__()
        self.embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=4)
        self.layer1 = nn.Sequential(
            nn.Linear(emb_dim, 4),
            nn.ReLU(),
            )
        self.layer2 = nn.Sequential(
            nn.Linear(emb_dim, 4),
            nn.ReLU(),
            )
        # self.layer3 = nn.Sequential(
        #     nn.Linear(12, 4),
        #     nn.ReLU(),
        #     nn.Linear(4, 1),
        #     nn.ReLU(),
        #     )
        self.p = nn.Parameter(torch.tensor([1.0]))
        self.layer1.apply(self.init_layer)
        self.layer2.apply(self.init_layer)
        # self.layer3.apply(self.init_layer)
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
    
    def init_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.zeros_(m.bias)

    def forward(self, ratings, item_ids, noise, init_user_emb):
        item_embs = self.embedding_item(item_ids) # (batch size, item size, reduce dim)
        noise = self.layer1(noise) # (batch size, reduce dim)
        user_emb = self.layer2(init_user_emb) # (batch size, reduce dim)
        noise = torch.einsum("ijk, ik->ij", item_embs, noise) # (batch size, item size)
        user_emb = torch.einsum("ijk, ik->ij", item_embs, user_emb) # (batch size, item size)
        denoise_output = self.p * ratings + noise + user_emb # (batch size, item size)
        return denoise_output

    def get_loss(self, true_preds, denoise_preds, mask=None):
        if mask is not None:
            loss = ((true_preds - denoise_preds) ** 2) * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = torch.mean((true_preds - denoise_preds) ** 2)
        reg_loss = self.embedding_item.weight.norm(2).pow(2)/self.num_items
        loss += reg_loss
        return loss