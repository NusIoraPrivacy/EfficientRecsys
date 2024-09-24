import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
import numpy as np
from utils.globals import *
from utils.util import get_noise_std
import copy
import random

class ClientsSampler(Dataset):
    def __init__(self, n):
        super().__init__()
        self.users_seq = np.arange(n)

    def __len__(self):
        return len(self.users_seq)

    def __getitem__(self, idx):
        return self.users_seq[idx]

class ClientsDataset(Dataset):
    def __init__(self, data_dict, n_items, n_users, n_user_feat, n_item_feat, args):
        self.data_dict = data_dict
        self.args = args
        # get rating vector
        if args.model == "MF" and args.als:
            self.rating_vector = np.zeros((n_users, n_items))
            for user_idx in self.data_dict:
                for record in self.data_dict[user_idx]:
                    item_idx, rating = record[0], record[1]
                    self.rating_vector[user_idx, item_idx] = rating
            self.c_vecs = 1 * (self.rating_vector > 0)
            self.rating_vector = sp.csr_matrix(self.rating_vector)
        self.n_user_feat = n_user_feat
        self.n_item_feat = n_item_feat

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        user_rating_list = self.data_dict[idx]
        user_rating_list = copy.deepcopy(user_rating_list)
        random.shuffle(user_rating_list)
        user_rating_list = user_rating_list[:self.args.n_sample_items]
        users = torch.tensor([idx] * len(user_rating_list)).to(self.args.device)
        items = torch.tensor([rate[0] for rate in user_rating_list]).to(self.args.device)
        ratings = torch.tensor([rate[1] for rate in user_rating_list]).to(self.args.device)
        rating_vec = None
        c_vec = None
        if self.args.model == "MF" and self.args.als:
            rating_vec = self.rating_vector[idx]
            c_vec = self.c_vecs[idx]
        item_feat = torch.tensor([rate[2:(2+self.n_item_feat)] for rate in user_rating_list])
        if self.n_user_feat > 0:
            user_feat = torch.tensor([rate[(-self.n_user_feat):] for rate in user_rating_list])
        else:
            user_feat = torch.tensor([])
        return users, items, ratings, rating_vec, c_vec, item_feat, user_feat

class CentralDataset(Dataset):
    def __init__(self, data_values, n_user_feat, n_item_feat, args):
        self.data_values = data_values
        self.args = args
        self.n_user_feat = n_user_feat
        self.n_item_feat = n_item_feat

    def __len__(self):
        return len(self.data_values)

    def __getitem__(self, idx):
        user_rating_list = self.data_values[idx]
        user = torch.tensor(user_rating_list[0], dtype=int)
        item = torch.tensor(user_rating_list[1], dtype=int)
        rating = torch.tensor(user_rating_list[2])
        item_feat = torch.tensor(user_rating_list[3:(3+self.n_item_feat)], dtype=int)
        if self.n_user_feat > 0:
            user_feat = torch.tensor(user_rating_list[(-self.n_user_feat):])
        else:
            user_feat = torch.tensor([])
        return user, item, rating, item_feat, user_feat

class DenoiseDataset(Dataset):
    def __init__(self, data_dict, base_model, n_users, n_items, n_user_feat, n_item_feat, args, max_item=150, noise=True, test=False):
        self.base_model = base_model
        self.data_dict = data_dict
        self.args = args
        self.n_users = n_users
        self.n_items = n_items
        if max_item is None:
            max_item = 0
            for user in self.data_dict:
                user_rating_list = self.data_dict[user]
                max_item = max(max_item, len(user_rating_list))
            self.max_item = max_item
        else:
            self.max_item = max_item
        self.n_user_feat = n_user_feat
        self.n_item_feat = n_item_feat
        self.sensitivity = norm_dict[args.dataset][args.model]
        self.noise = noise
        self.test = test

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        # prepare ids and ratings
        try:
            user_rating_list = self.data_dict[idx]
        except:
            return self.__getitem__(idx+1)
        user_rating_list = user_rating_list.copy()
        random.shuffle(user_rating_list)
        user_ids = torch.tensor([idx] * self.max_item).to(self.args.device)
        item_ids = [rate[0] for rate in user_rating_list][:self.max_item] + [-1] * (self.max_item-len(user_rating_list))
        item_ids = torch.tensor(item_ids).to(self.args.device)
        masks = (item_ids >= 0).float()
        item_ids[item_ids==-1] = 0
        ratings = [rate[1] for rate in user_rating_list][:self.max_item] + [0] * (self.max_item-len(user_rating_list))
        ratings = torch.tensor(ratings).to(self.args.device)
        item_feat = [rate[2:(2+self.n_item_feat)] for rate in user_rating_list][:self.max_item] + [[0]*self.n_item_feat] * (self.max_item-len(user_rating_list))
        item_feat = torch.tensor(item_feat).to(self.args.device) # (bs, item_size, feat num)
        if self.n_user_feat > 0:
            user_feat = [rate[(-self.n_user_feat):] for rate in user_rating_list][:self.max_item] + [[0]*self.n_user_feat] * (self.max_item-len(user_rating_list))
            user_feat = torch.tensor(user_feat).to(self.args.device)
        else:
            user_feat = torch.tensor([]).to(self.args.device)
        if self.noise:
            sigma = get_noise_std(self.sensitivity, self.args)
            if self.test:
                with torch.no_grad():
                    predictions, noises, init_user_embs = self.base_model(user_ids, item_ids, user_feats=user_feat, item_feats=item_feat, noise_std=sigma)
            else:
                predictions, noises, init_user_embs = self.base_model(user_ids, item_ids, user_feats=user_feat, item_feats=item_feat, noise_std=sigma)
            noises, init_user_embs = noises[0], init_user_embs[0]
        else:
            if self.test:
                with torch.no_grad():
                    predictions = self.base_model(user_ids, item_ids, user_feats=user_feat, item_feats=item_feat)
            else:
                predictions = self.base_model(user_ids, item_ids, user_feats=user_feat, item_feats=item_feat)
            noises, init_user_embs = torch.tensor([]), torch.tensor([])
        return predictions, noises,  init_user_embs, item_ids, ratings, masks, item_feat
