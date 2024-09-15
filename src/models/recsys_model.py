import torch
from torch import nn
from utils.globals import *
import copy
import numpy as np

class MF(nn.Module):
    def __init__(self, num_users, num_items, args, **kwargs):
        super(MF, self).__init__()
        self.embedding_user = nn.Embedding(
            num_embeddings=num_users, embedding_dim=args.n_factors)
        self.embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=args.n_factors)
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(num_items))
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.max_norm = norm_dict["MF"]
        nn.init.kaiming_normal_(self.embedding_user.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_item.weight, mode='fan_out')

    def forward(self, users, items, noise_std=0, users_emb=None, **kwargs):
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
        x_biases = torch.cat([self.user_bias[users].unsqueeze(-1), self.item_bias[items].unsqueeze(-1)], dim=-1).sum(1)
        predictions += x_biases
        if noise_std > 0:
            return predictions, noise, init_user_emb
        return predictions

    def get_loss(self, ratings, predictions):
        loss = torch.mean((ratings - predictions) ** 2)
        # reg_loss = self.embedding_user.weight.norm(2).pow(2)/self.num_users
        # loss += reg_loss
        return loss

# neural collaborative filtering
class NCF(nn.Module):
    def __init__(self, num_users, num_items, args, **kwargs):
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
            nn.Linear(args.n_factors*3, 1, bias=False),
            # nn.ReLU()
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
            if m.bias is not None:
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

    def forward(self, users, items, noise_std=0, users_emb=None, **kwargs):
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
        if noise_std > 0:
            noises = torch.cat([gmf_noise, ncf_noise], dim=-1)
            init_users_emb = torch.cat([init_gmf_users_emb, init_ncf_users_emb], dim=-1)
            return predictions, noises, init_users_emb
        return predictions

    def get_loss(self, ratings, predictions):
        loss = torch.mean((ratings - predictions) ** 2)
        # reg_loss = (self.gmf_embedding_user.weight.norm(2).pow(2)+self.ncf_embedding_user.weight.norm(2).pow(2))/self.num_users
        # loss += reg_loss
        return loss

class FM(nn.Module):
    def __init__(self, num_users, num_items, num_user_feats, num_item_feats, args):
        super().__init__()
        self.embedding_user = nn.Embedding(
            num_embeddings=num_users, embedding_dim=args.n_factors)
        self.embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=args.n_factors)
        self.embedding_user_feats = nn.Embedding(
            num_embeddings=num_user_feats, embedding_dim=args.n_factors)
        self.embedding_item_feats = nn.Embedding(
            num_embeddings=num_item_feats, embedding_dim=args.n_factors)
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(num_items))
        self.user_feat_bias = nn.Parameter(torch.zeros(num_user_feats))
        self.item_feat_bias = nn.Parameter(torch.zeros(num_item_feats))
        self.offset = nn.Parameter(torch.zeros(1))
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.num_user_feats = num_user_feats
        self.num_item_feats = num_item_feats
        self.max_norm = norm_dict["FM"]
        self.n_max_user_feat = n_max_user_feat_dict[args.dataset]
        nn.init.kaiming_normal_(self.embedding_user.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_item.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_user_feats.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_item_feats.weight, mode='fan_out')
    
    def add_noise(self, users_emb, noise_std, max_norm, user_feat=False):
        init_user_emb = copy.deepcopy(users_emb.data) # (item_size, n_features, dim)
        if user_feat:
            zero_embs = torch.sum(init_user_emb == 0, dim=-1)
            zero_embs = zero_embs.unsqueeze(-1)
        all_norms = torch.norm(users_emb, p=2, dim=-1)
        users_emb = users_emb * torch.clamp(max_norm / all_norms, max=1).unsqueeze(-1)
        if user_feat:
            noise = torch.normal(mean=0., std=noise_std, size=users_emb.shape, device=users_emb.device)
            noise = noise * zero_embs
        else:
            noise = torch.normal(mean=0., std=noise_std, size=(users_emb.shape[-1],), device=users_emb.device)
            noise = noise.repeat(users_emb.shape[0], 1, 1)
        users_emb += noise
        all_norms = torch.norm(users_emb, p=2, dim=-1)
        users_emb = users_emb * torch.clamp(max_norm / all_norms, max=1).unsqueeze(-1)
        noise = users_emb - init_user_emb
        return users_emb, noise, init_user_emb

    def forward(self, user_ids, item_ids, user_feats, item_feats, noise_std=0):
        user_ids = user_ids.unsqueeze(dim=1) # (item_size, 1)
        user_embs = self.embedding_user(user_ids) # (item_size, 1, emb_dim)
        if noise_std > 0:
            user_embs, user_emb_noises, init_user_embs = self.add_noise(user_embs, noise_std, self.max_norm[0])
        user_feat_embs = torch.einsum("ij, ki->kij", self.embedding_user_feats.weight, user_feats) # (item_size, num_feats, emb_dim)
        if noise_std > 0:
            user_feat_embs, user_feat_emb_noises, init_user_feat_embs = self.add_noise(user_feat_embs, noise_std, self.max_norm[1]/np.sqrt(self.n_max_user_feat), user_feat=True)
        item_ids = item_ids.unsqueeze(dim=1) # (item_size, 1)
        item_embs = self.embedding_item(item_ids) # (item_size, 1, emb_dim)
        item_feat_embs = torch.einsum("ij, ki->kij", self.embedding_item_feats.weight, item_feats) # (item_size, num_feats, emb_dim)
        all_embs = torch.cat([user_embs, item_embs, user_feat_embs, item_feat_embs], dim=1) # (item_size, num_feats, emb_dim)
        pow_of_sum = all_embs.sum(dim=1).pow(2) # (item_size, emb_dim)
        sum_of_pow = all_embs.pow(2).sum(dim=1) # (item_size, emb_dim)
        fm_out = (pow_of_sum - sum_of_pow).sum(dim=-1)*0.5  # item_size
        feat_bias = torch.cat([torch.einsum("i, ji->ji", self.user_feat_bias, user_feats), torch.einsum("i, ji->ji", self.item_feat_bias, item_feats)], dim=-1)
        x_biases = torch.cat([self.user_bias[user_ids], self.item_bias[item_ids], feat_bias], dim=-1).sum(1) # item_size
        fm_out +=  x_biases + self.offset # item_size
        if noise_std > 0:
            noises = torch.cat([user_emb_noises, user_feat_emb_noises], dim=1)
            init_users_emb = torch.cat([init_user_embs, init_user_feat_embs], dim=1)
            return fm_out, noises, init_users_emb
        return fm_out

    def get_loss(self, ratings, predictions):
        loss = torch.mean((ratings - predictions) ** 2)
        # reg_loss = self.embedding_user.weight.norm(2).pow(2)/self.num_users
        # loss += reg_loss
        return loss

class DeepFM(nn.Module):
    def __init__(self, num_users, num_items, num_user_feats, num_item_feats, args):
        super().__init__()
        self.embedding_user = nn.Embedding(
            num_embeddings=num_users, embedding_dim=args.n_factors)
        self.embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=args.n_factors)
        self.embedding_user_feats = nn.Embedding(
            num_embeddings=num_user_feats, embedding_dim=args.n_factors)
        self.embedding_item_feats = nn.Embedding(
            num_embeddings=num_item_feats, embedding_dim=args.n_factors)
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(num_items))
        self.user_feat_bias = nn.Parameter(torch.zeros(num_user_feats))
        self.item_feat_bias = nn.Parameter(torch.zeros(num_item_feats))
        self.offset = nn.Parameter(torch.zeros(1))
        self.embed_output_dim = (2 + num_user_feats + num_item_feats) * args.n_factors
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_output_dim, args.n_factors*4),
            nn.BatchNorm1d(args.n_factors*4),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(args.n_factors*4, args.n_factors*2),
            nn.BatchNorm1d(args.n_factors*2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(args.n_factors*2, 1)
            )
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.num_user_feats = num_user_feats
        self.num_item_feats = num_item_feats
        self.max_norm = norm_dict["FM"]
        self.n_max_user_feat = n_max_user_feat_dict[args.dataset]
        nn.init.kaiming_normal_(self.embedding_user.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_item.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_user_feats.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_item_feats.weight, mode='fan_out')

    def add_noise(self, users_emb, noise_std, max_norm, user_feat=False):
        init_user_emb = copy.deepcopy(users_emb.data) # (item_size, n_features, dim)
        if user_feat:
            zero_embs = torch.sum(init_user_emb == 0, dim=-1)
            zero_embs = zero_embs.unsqueeze(-1)
        all_norms = torch.norm(users_emb, p=2, dim=-1)
        users_emb = users_emb * torch.clamp(max_norm / all_norms, max=1).unsqueeze(-1)
        if user_feat:
            noise = torch.normal(mean=0., std=noise_std, size=users_emb.shape, device=users_emb.device)
            noise = noise * zero_embs
        else:
            noise = torch.normal(mean=0., std=noise_std, size=(users_emb.shape[-1],), device=users_emb.device)
            noise = noise.repeat(users_emb.shape[0], 1, 1)
        users_emb += noise
        all_norms = torch.norm(users_emb, p=2, dim=-1)
        users_emb = users_emb * torch.clamp(max_norm / all_norms, max=1).unsqueeze(-1)
        noise = users_emb - init_user_emb
        return users_emb, noise, init_user_emb
        
    def forward(self, user_ids, item_ids, user_feats, item_feats, noise_std=0):
        user_ids = user_ids.unsqueeze(dim=1) # (item_size, 1)
        user_embs = self.embedding_user(user_ids) # (item_size, 1, emb_dim)
        if noise_std > 0:
            user_embs, user_emb_noises, init_user_embs = self.add_noise(user_embs, noise_std, self.max_norm[0])
        user_feat_embs = torch.einsum("ij, ki->kij", self.embedding_user_feats.weight, user_feats) # (item_size, num_feats, emb_dim)
        if noise_std > 0:
            user_feat_embs, user_feat_emb_noises, init_user_feat_embs = self.add_noise(user_feat_embs, noise_std, self.max_norm[1]/np.sqrt(self.n_max_user_feat), user_feat=True)
        item_ids = item_ids.unsqueeze(dim=1) # (item_size, 1)
        item_embs = self.embedding_item(item_ids) # (item_size, 1, emb_dim)
        item_feat_embs = torch.einsum("ij, ki->kij", self.embedding_item_feats.weight, item_feats) # (item_size, num_feats, emb_dim)
        all_embs = torch.cat([user_embs, item_embs, user_feat_embs, item_feat_embs], dim=1) # (item_size, num_feats, emb_dim)
        pow_of_sum = all_embs.sum(dim=1).pow(2) # (item_size, emb_dim)
        sum_of_pow = all_embs.pow(2).sum(dim=1) # (item_size, emb_dim)
        fm_out = (pow_of_sum - sum_of_pow).sum(dim=-1)*0.5  # item_size
        feat_bias = torch.cat([torch.einsum("i, ji->ji", self.user_feat_bias, user_feats), torch.einsum("i, ji->ji", self.item_feat_bias, item_feats)], dim=-1)
        x_biases = torch.cat([self.user_bias[user_ids], self.item_bias[item_ids], feat_bias], dim=-1).sum(1) # item_size
        fm_out +=  x_biases + self.offset # item_size
        all_embs = all_embs.view(-1, self.embed_output_dim) # (item_size, num_feats*emb_dim)
        dnn_out = self.mlp(all_embs).squeeze(-1)
        predictions = fm_out + dnn_out
        if noise_std > 0:
            noises = torch.cat([user_emb_noises, user_feat_emb_noises], dim=1)
            init_users_emb = torch.cat([init_user_embs, init_user_feat_embs], dim=1)
            return predictions, noises, init_users_emb
        return predictions

    def get_loss(self, ratings, predictions):
        loss = torch.mean((ratings - predictions) ** 2)
        # reg_loss = self.embedding_user.weight.norm(2).pow(2)/self.num_users
        # loss += reg_loss
        return loss