import torch
from torch import nn
from utils.globals import *
import copy
import numpy as np
import math

class MF(nn.Module):
    def __init__(self, num_users, num_items, args, max_norm=None, **kwargs):
        super(MF, self).__init__()
        self.embedding_user = nn.Embedding(
            num_embeddings=num_users, embedding_dim=args.n_factors)
        self.embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=args.n_factors).requires_grad_(False)
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(num_items))
        self.A = nn.Parameter(torch.zeros(num_items, args.rank))
        self.B = torch.zeros(args.rank, args.n_factors).to(args.device)
        nn.init.normal_(self.B, mean=0, std=1*math.sqrt(1/args.rank))
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        if max_norm is None:
            self.max_norm = norm_dict[args.dataset]["MF"]
        else:
            self.max_norm = max_norm
        nn.init.kaiming_normal_(self.embedding_user.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_item.weight, mode='fan_out')
    
    def update_params(self):
        self.embedding_item.weight.data = self.A @ self.B + self.embedding_item.weight
        nn.init.normal_(self.B, mean=0, std=1*math.sqrt(1/self.args.rank))
    
    def reset_A(self):
        nn.init.zeros_(self.A)

    def forward(self, users, items, train=True, **kwargs):
        # compute embedding
        users_emb = self.embedding_user(users) # item_size, dim
        if train:
            item_emb_mat = self.A @ self.B + self.embedding_item.weight
            items_emb = item_emb_mat[items]
        else:
            items_emb = self.embedding_item(items)
        inner_pro = torch.mul(users_emb, items_emb)
        predictions = torch.sum(inner_pro, dim=1)
        user_biases = self.user_bias[users].unsqueeze(-1)
        x_biases = torch.cat([user_biases, self.item_bias[items].unsqueeze(-1)], dim=-1).sum(1) # item_size
        predictions += x_biases
        return predictions

    def get_loss(self, ratings, predictions):
        loss = torch.mean((ratings - predictions) ** 2)
        if self.args.regularization:
            item_emb_mat = self.A @ self.B + self.embedding_item.weight
            reg_loss = self.embedding_user.weight.norm(2).pow(2) * self.args.l2_reg_u + item_emb_mat.norm(2).pow(2) * self.args.l2_reg_i
            loss += reg_loss
        return loss
    
    def get_loss_central(self, ratings, predictions, dp=False):
        if dp:
            loss = (ratings - predictions) ** 2
            loss = loss/len(ratings)
        else:
            loss = torch.mean((ratings - predictions) ** 2)
            if self.args.regularization:
                reg_loss = self.embedding_user.weight.norm(2).pow(2) * self.args.l2_reg_u + self.embedding_item.weight.norm(2).pow(2) * self.args.l2_reg_i
                loss += reg_loss
        return loss

# neural collaborative filtering
class NCF(nn.Module):
    def __init__(self, num_users, num_items, args, max_norm=None, **kwargs):
        super(NCF, self).__init__()
        self.gmf_embedding_user = nn.Embedding(
            num_embeddings=num_users, embedding_dim=args.n_factors * 2)
        self.ncf_embedding_user = nn.Embedding(
            num_embeddings=num_users, embedding_dim=args.n_factors * 2)
        self.gmf_embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=args.n_factors * 2)
        self.ncf_embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=args.n_factors * 2)
        self.gmf_A = nn.Parameter(torch.zeros(num_items, args.rank))
        self.ncf_A = nn.Parameter(torch.zeros(num_items, args.rank))
        self.gmf_B = torch.zeros(args.rank, args.n_factors * 2).to(args.device)
        self.ncf_B = torch.zeros(args.rank, args.n_factors * 2).to(args.device)
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
        if max_norm is None:
            self.max_norm = norm_dict[args.dataset]["NCF"]
        else:
            self.max_norm = max_norm
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

    def update_params(self):
        self.gmf_embedding_item.weight.data = self.gmf_A @ self.gmf_B + self.gmf_embedding_item.weight
        self.ncf_embedding_item.weight.data = self.ncf_A @ self.ncf_B + self.ncf_embedding_item.weight
        nn.init.normal_(self.gmf_B, mean=0, std=1*math.sqrt(1/self.args.rank))
        nn.init.normal_(self.ncf_B, mean=0, std=1*math.sqrt(1/self.args.rank))
    
    def reset_A(self):
        nn.init.zeros_(self.gmf_A)
        nn.init.zeros_(self.ncf_A)

    def forward(self, users, items, train=True, **kwargs):
        # compute embedding
        gmf_users_emb = self.gmf_embedding_user(users)
        ncf_users_emb = self.ncf_embedding_user(users)
        if train:
            gmf_item_emb_mat = self.gmf_A @ self.gmf_B + self.gmf_embedding_item.weight
            gmf_items_emb = gmf_item_emb_mat[items]
            ncf_item_emb_mat = self.ncf_A @ self.ncf_B + self.ncf_embedding_item.weight
            ncf_items_emb = ncf_item_emb_mat[items]
        else:
            gmf_items_emb = self.gmf_embedding_item(items)
            ncf_items_emb = self.ncf_embedding_item(items)
        all_ncf_embs = torch.cat([ncf_users_emb, ncf_items_emb], dim=1)
        ncf_h = self.ncf_hidden_layers(all_ncf_embs)
        gmf_h = gmf_users_emb * gmf_items_emb
        h = torch.cat([gmf_h, ncf_h], 1)
        predictions = self.output_layer(h)
        predictions = predictions.squeeze(dim=-1)
        return predictions

    def get_loss(self, ratings, predictions):
        loss = torch.mean((ratings - predictions) ** 2)
        if self.args.regularization:
            reg_loss = (self.gmf_embedding_user.weight.norm(2).pow(2)+self.ncf_embedding_user.weight.norm(2).pow(2)) * self.args.l2_reg_u
            loss += reg_loss
        return loss
    
    def get_loss_central(self, ratings, predictions, dp=False):
        if dp:
            loss = (ratings - predictions) ** 2
            loss = loss/len(ratings)
        else:
            loss = torch.mean((ratings - predictions) ** 2)
            if self.args.regularization:
                reg_loss = (self.gmf_embedding_user.weight.norm(2).pow(2)+self.ncf_embedding_user.weight.norm(2).pow(2)) * self.args.l2_reg_u
                reg_loss += (self.gmf_embedding_item.weight.norm(2).pow(2)+self.ncf_embedding_item.weight.norm(2).pow(2)) * self.args.l2_reg_i
                loss += reg_loss
        return loss

class FM(nn.Module):
    def __init__(self, num_users, num_items, num_user_feats, num_item_feats, args, max_norm=None):
        super().__init__()
        self.embedding_user = nn.Embedding(
            num_embeddings=num_users, embedding_dim=args.n_factors)
        self.embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=args.n_factors)
        self.embedding_item_feats = nn.Embedding(
            num_embeddings=num_item_feats, embedding_dim=args.n_factors)
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(num_items))
        self.item_feat_bias = nn.Parameter(torch.zeros(num_item_feats))
        self.offset = nn.Parameter(torch.zeros(1))
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.num_user_feats = num_user_feats
        self.num_item_feats = num_item_feats
        if max_norm is None:
            self.max_norm = norm_dict[args.dataset]["FM"]
        else:
            self.max_norm = max_norm
        self.n_max_user_feat = n_max_user_feat_dict[args.dataset]
        nn.init.kaiming_normal_(self.embedding_user.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_item.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_item_feats.weight, mode='fan_out')
        if num_user_feats > 0:
            self.embedding_user_feats = nn.Embedding(
                num_embeddings=num_user_feats, embedding_dim=args.n_factors)
            self.user_feat_bias = nn.Parameter(torch.zeros(num_user_feats))
            nn.init.kaiming_normal_(self.embedding_user_feats.weight, mode='fan_out')
        self.A = nn.Parameter(torch.zeros(num_items, args.rank))
        self.B = torch.zeros(args.rank, args.n_factors).to(args.device)

    def update_params(self):
        self.embedding_item.weight.data = self.A @ self.B + self.embedding_item.weight
        nn.init.normal_(self.B, mean=0, std=1*math.sqrt(1/self.args.rank))
    
    def reset_A(self):
        nn.init.zeros_(self.A)
    
    def forward(self, user_ids, item_ids, user_feats, item_feats, train=True, **kwargs):
        user_ids = user_ids.unsqueeze(dim=1) # (item_size, 1)
        user_embs = self.embedding_user(user_ids) # (item_size, 1, emb_dim)
        if self.num_user_feats > 0:
            user_feat_embs = torch.einsum("ij, ki->kij", self.embedding_user_feats.weight, user_feats) # (item_size, num_feats, emb_dim)
        item_ids = item_ids.unsqueeze(dim=1) # (item_size, 1)

        if train:
            item_emb_mat = self.A @ self.B + self.embedding_item.weight
            item_embs = item_emb_mat[item_ids]
        else:
            item_embs = self.embedding_item(item_ids)
        item_feat_embs = torch.einsum("ij, ki->kij", self.embedding_item_feats.weight, item_feats) # (item_size, num_feats, emb_dim)
        if self.num_user_feats > 0:
            all_embs = torch.cat([user_embs, item_embs, user_feat_embs, item_feat_embs], dim=1) # (item_size, num_feats, emb_dim)
        else:
            all_embs = torch.cat([user_embs, item_embs, item_feat_embs], dim=1)
        pow_of_sum = all_embs.sum(dim=1).pow(2) # (item_size, emb_dim)
        sum_of_pow = all_embs.pow(2).sum(dim=1) # (item_size, emb_dim)
        fm_out = (pow_of_sum - sum_of_pow).sum(dim=-1)*0.5  # item_size
        if self.num_user_feats > 0:
            feat_bias = torch.cat([torch.einsum("i, ji->ji", self.user_feat_bias, user_feats), torch.einsum("i, ji->ji", self.item_feat_bias, item_feats)], dim=-1)
        else:
            feat_bias = torch.einsum("i, ji->ji", self.item_feat_bias, item_feats)

        user_biases = self.user_bias[user_ids] # item_size, 1
        x_biases = torch.cat([user_biases, self.item_bias[item_ids], feat_bias], dim=-1).sum(1) # item_size
        fm_out +=  x_biases + self.offset # item_size
        return fm_out

    def get_loss(self, ratings, predictions):
        loss = torch.mean((ratings - predictions) ** 2)
        if self.args.regularization:
            item_emb_mat = self.A @ self.B + self.embedding_item.weight
            reg_loss = self.embedding_user.weight.norm(2).pow(2) * self.args.l2_reg_u + item_emb_mat.norm(2).pow(2) * self.args.l2_reg_i
            loss += reg_loss
        return loss
    
    def get_loss_central(self, ratings, predictions, dp=False):
        if dp:
            loss = (ratings - predictions) ** 2
            loss = loss/len(ratings)
        else:
            loss = torch.mean((ratings - predictions) ** 2)
            if self.args.regularization:
                reg_loss = self.embedding_user.weight.norm(2).pow(2) * self.args.l2_reg_u
                reg_loss += (self.embedding_item.weight.norm(2).pow(2)+self.embedding_item_feats.weight.norm(2).pow(2)) * self.args.l2_reg_i
                if self.num_user_feats > 0:
                    reg_loss += self.embedding_user_feats.weight.norm(2).pow(2) * self.args.l2_reg_u
                loss += reg_loss
        return loss

class DeepFM(nn.Module):
    def __init__(self, num_users, num_items, num_user_feats, num_item_feats, args, max_norm=None):
        super().__init__()
        self.embedding_user = nn.Embedding(
            num_embeddings=num_users, embedding_dim=args.n_factors)
        self.embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=args.n_factors)
        
        self.embedding_item_feats = nn.Embedding(
            num_embeddings=num_item_feats, embedding_dim=args.n_factors)
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(num_items))
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
        if max_norm is None:
            self.max_norm = norm_dict[args.dataset]["DeepFM"]
        else:
            self.max_norm = max_norm
        self.n_max_user_feat = n_max_user_feat_dict[args.dataset]
        nn.init.kaiming_normal_(self.embedding_user.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_item.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_item_feats.weight, mode='fan_out')
        self.A = nn.Parameter(torch.zeros(num_items, args.rank))
        self.B = torch.zeros(args.rank, args.n_factors).to(args.device)

        if num_user_feats > 0:
            self.embedding_user_feats = nn.Embedding(
                num_embeddings=num_user_feats, embedding_dim=args.n_factors)
            self.user_feat_bias = nn.Parameter(torch.zeros(num_user_feats))
            nn.init.kaiming_normal_(self.embedding_user_feats.weight, mode='fan_out')
    
    def update_params(self):
        self.embedding_item.weight.data = self.A @ self.B + self.embedding_item.weight
        nn.init.normal_(self.B, mean=0, std=1*math.sqrt(1/self.args.rank))
    
    def reset_A(self):
        nn.init.zeros_(self.A)

    def forward(self, user_ids, item_ids, user_feats, item_feats, train=True, **kwargs):
        user_ids = user_ids.unsqueeze(dim=1) # (item_size, 1)
        user_embs = self.embedding_user(user_ids) # (item_size, 1, emb_dim)
        if self.num_user_feats > 0:
            user_feat_embs = torch.einsum("ij, ki->kij", self.embedding_user_feats.weight, user_feats) # (item_size, num_feats, emb_dim)
        item_ids = item_ids.unsqueeze(dim=1) # (item_size, 1)
        if train:
            item_emb_mat = self.A @ self.B + self.embedding_item.weight
            item_embs = item_emb_mat[item_ids]
        else:
            item_embs = self.embedding_item(item_ids) # (item_size, 1, emb_dim)
        item_feat_embs = torch.einsum("ij, ki->kij", self.embedding_item_feats.weight, item_feats) # (item_size, num_feats, emb_dim)
        if self.num_user_feats > 0:
            all_embs = torch.cat([user_embs, item_embs, user_feat_embs, item_feat_embs], dim=1) # (item_size, num_feats, emb_dim)
        else:
            all_embs = torch.cat([user_embs, item_embs, item_feat_embs], dim=1) # (item_size, num_feats, emb_dim)
        pow_of_sum = all_embs.sum(dim=1).pow(2) # (item_size, emb_dim)
        sum_of_pow = all_embs.pow(2).sum(dim=1) # (item_size, emb_dim)
        fm_out = (pow_of_sum - sum_of_pow).sum(dim=-1)*0.5  # item_size
        if self.num_user_feats > 0:
            feat_bias = torch.cat([torch.einsum("i, ji->ji", self.user_feat_bias, user_feats), torch.einsum("i, ji->ji", self.item_feat_bias, item_feats)], dim=-1)
        else:
            feat_bias = torch.einsum("i, ji->ji", self.item_feat_bias, item_feats)
        user_biases = self.user_bias[user_ids] # item_size, 1
        x_biases = torch.cat([user_biases, self.item_bias[item_ids], feat_bias], dim=-1).sum(1) # item_size
        fm_out +=  x_biases + self.offset # item_size
        all_embs = all_embs.view(-1, self.embed_output_dim) # (item_size, num_feats*emb_dim)
        dnn_out = self.mlp(all_embs).squeeze(-1)
        predictions = fm_out + dnn_out
        return predictions

    def get_loss(self, ratings, predictions):
        loss = torch.mean((ratings - predictions) ** 2)
        if self.args.regularization:
            item_emb_mat = self.A @ self.B + self.embedding_item.weight
            reg_loss = self.embedding_user.weight.norm(2).pow(2) * self.args.l2_reg_u + item_emb_mat.norm(2).pow(2) * self.args.l2_reg_i
            loss += reg_loss
        return loss
    
    def get_loss_central(self, ratings, predictions, dp=False):
        if dp:
            loss = (ratings - predictions) ** 2
            loss = loss/len(ratings)
        else:
            loss = torch.mean((ratings - predictions) ** 2)
            if self.args.regularization:
                reg_loss = self.embedding_user.weight.norm(2).pow(2) * self.args.l2_reg_u
                reg_loss += (self.embedding_item.weight.norm(2).pow(2)+self.embedding_item_feats.weight.norm(2).pow(2)) * self.args.l2_reg_i
                if self.num_user_feats > 0:
                    reg_loss += self.embedding_user_feats.weight.norm(2).pow(2) * self.args.l2_reg_u
                loss += reg_loss
        return loss