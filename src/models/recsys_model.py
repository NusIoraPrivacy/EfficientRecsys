import torch
from torch import nn
from utils.globals import *
import copy

class MF(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(MF, self).__init__()
        self.embedding_user = nn.Embedding(
            num_embeddings=num_users, embedding_dim=args.n_factors)
        self.embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=args.n_factors)
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.max_norm = norm_dict["MF"]
        nn.init.kaiming_normal_(self.embedding_user.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_item.weight, mode='fan_out')

    def forward(self, users, items, noise_std=0, users_emb=None):
        # compute embedding
        if users_emb is None:
            users_emb = self.embedding_user(users)
        if noise_std > 0:
            init_user_emb = copy.deepcopy(users_emb.data)
            all_norms = torch.norm(users_emb, p=2, dim=-1)
            users_emb = users_emb * torch.clamp(self.max_norm / all_norms, max=1).unsqueeze(-1)
            noise = torch.normal(mean=0., std=noise_std, size=(users_emb.shape[-1],), device=users_emb.device)
            noise = noise.repeat(users_emb.shape[0], 1)
            users_emb += noise
            all_norms = torch.norm(users_emb, p=2, dim=-1)
            users_emb = users_emb * torch.clamp(self.max_norm / all_norms, max=1).unsqueeze(-1)
            noise = users_emb - init_user_emb
        items_emb = self.embedding_item(items)
        inner_pro = torch.mul(users_emb, items_emb)
        predictions = torch.sum(inner_pro, dim=1)
        if noise_std > 0:
            return predictions, noise, init_user_emb
        return predictions

    def get_loss(self, ratings, predictions):
        loss = torch.mean((ratings - predictions) ** 2)
        reg_loss = self.embedding_user.weight.norm(2).pow(2)/self.num_users
        loss += reg_loss
        return loss

# neural collaborative filtering
class NCF(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(NCF, self).__init__()
        self.gmf_embedding_user = nn.Embedding(
            num_embeddings=num_users, embedding_dim=args.n_factors * 2)
        self.ncf_embedding_user = nn.Embedding(
            num_embeddings=num_users, embedding_dim=args.n_factors * 2)
        self.gmf_embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=args.n_factors * 2)
        self.ncf_embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=args.n_factors * 2)
        self.ncf_hidden_layers = nn.Sequential(
            nn.Linear(args.n_factors * 4, args.n_factors * 2),
            nn.ReLU(),
            nn.Linear(args.n_factors * 2, args.n_factors),
            nn.ReLU(),
            )
        self.output_layer = nn.Sequential(
            nn.Linear(args.n_factors*3, 1),
            nn.ReLU()
            )
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.max_norm = norm_dict["NCF"]
        self.init_embedding()
        self.ncf_hidden_layers.apply(self.init_layer)
        self.output_layer.apply(self.init_layer)
    
    def init_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.zeros_(m.bias)

    def init_embedding(self):
        nn.init.kaiming_normal_(self.gmf_embedding_user.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.ncf_embedding_user.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.gmf_embedding_item.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.ncf_embedding_item.weight, mode='fan_out')

    def add_noise(self, users_emb, noise_std, max_norm):
        init_user_emb = copy.deepcopy(users_emb.data)
        all_norms = torch.norm(users_emb, p=2, dim=-1)
        users_emb = users_emb * torch.clamp(max_norm / all_norms, max=1).unsqueeze(-1)
        noise = torch.normal(mean=0., std=noise_std, size=(users_emb.shape[-1],), device=users_emb.device)
        noise = noise.repeat(users_emb.shape[0], 1)
        users_emb += noise
        all_norms = torch.norm(users_emb, p=2, dim=-1)
        users_emb = users_emb * torch.clamp(max_norm / all_norms, max=1).unsqueeze(-1)
        # print(users_emb)
        noise = users_emb - init_user_emb
        return users_emb, noise, init_user_emb

    def forward(self, users, items, noise_std=0, users_emb=None):
        # compute embedding
        if users_emb is None:
            gmf_users_emb = self.gmf_embedding_user(users)
            ncf_users_emb = self.ncf_embedding_user(users)
            # print(users_emb)
        if noise_std > 0:
            gmf_users_emb, gmf_noise, init_gmf_users_emb = self.add_noise(gmf_users_emb, noise_std, self.max_norm[0])
            ncf_users_emb, ncf_noise, init_ncf_users_emb = self.add_noise(ncf_users_emb, noise_std, self.max_norm[1])
        gmf_items_emb = self.gmf_embedding_item(items)
        ncf_items_emb = self.ncf_embedding_item(items)
        all_ncf_embs = torch.cat([ncf_users_emb, ncf_items_emb], dim=1)
        ncf_h = self.ncf_hidden_layers(all_ncf_embs)
        gmf_h = gmf_users_emb * gmf_items_emb
        h = torch.cat([gmf_h, ncf_h], 1)
        predictions = self.output_layer(h)
        predictions = predictions.squeeze(dim=-1)
        noises = torch.cat([gmf_noise, ncf_noise], dim=-1)
        init_users_emb = torch.cat([init_gmf_users_emb, init_ncf_users_emb], dim=-1)
        if noise_std > 0:
            return predictions, noises, init_users_emb
        return predictions

    def get_loss(self, ratings, predictions):
        loss = torch.mean((ratings - predictions) ** 2)
        reg_loss = (self.gmf_embedding_user.weight.norm(2).pow(2)+self.ncf_embedding_user.weight.norm(2).pow(2))/self.num_users
        loss += reg_loss
        return loss