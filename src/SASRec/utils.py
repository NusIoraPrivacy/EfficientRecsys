import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import argparse
from utils.util import str2bool, str2type
from utils.globals import pos_thd
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

def get_args():
    # python
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument("--early_stop", type=int, default=20, 
                        help = "number of rounds/patience for early stop")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--compress", type=str, default="svd", choices=["none", "svd", "ternquant", "8intquant", "colr"])
    args = parser.parse_args()
    return args

def load_data(args):
    # rating_path = f"{args.root_path}/data/{args.dataset}/ratings_Beauty.csv"
    if args.dataset == "amazon-beauty":
        rating_path = f"{args.root_path}/data/amazon/ratings_Beauty.csv"
        rating_df = pd.read_csv(rating_path, header=None, names=["UserID", "ItemID", "Rating", "TimeStamp"], encoding="iso-8859-1")
    elif args.dataset == "amazon-sport":
        rating_path = f"{args.root_path}/data/amazon/ratings_Sports_and_Outdoors.csv"
        rating_df = pd.read_csv(rating_path, header=None, names=["UserID", "ItemID", "Rating", "TimeStamp"], encoding="iso-8859-1")
    elif args.dataset == "amazon-game":
        rating_path = f"{args.root_path}/data/amazon/ratings_Toys_and_Games.csv"
        rating_df = pd.read_csv(rating_path, header=None, names=["UserID", "ItemID", "Rating", "TimeStamp"], encoding="iso-8859-1")
    elif args.dataset == "ml-1m":
        rating_path = f"{args.root_path}/data/{args.dataset}/ratings.dat"
        rating_df = pd.read_csv(rating_path, delimiter='::', header=None, names=["UserID", "ItemID", "Rating", "TimeStamp"])
    
    thd = pos_thd[args.dataset]
    rating_df = rating_df[rating_df["Rating"] >= thd]
    # standard id
    itemIDs = rating_df.ItemID.unique()
    itemid2encode = {}
    for i, itemid in enumerate(itemIDs):
        itemid2encode[itemid] = i
    rating_df['ItemID'] = rating_df['ItemID'].apply(lambda x: itemid2encode[x])
    # list items per user
    n_items = len(itemid2encode)
    rating_df = rating_df.sort_values(by='TimeStamp', ascending=True)
    rated_df = rating_df.groupby('UserID').agg({'ItemID':lambda x: list(x)})
    rated_df = rated_df.reset_index()
    rated_df["n"] = rated_df["ItemID"].apply(lambda x: len(x))
    rated_df = rated_df[rated_df["n"]>=3]
    # standard id
    userIDs = rated_df.UserID.unique()
    userid2encode = {}
    for i, userid in enumerate(userIDs):
        userid2encode[userid] = i
    rated_df['UserID'] = rated_df['UserID'].apply(lambda x: userid2encode[x])
    n_users = len(userid2encode)
    # train test split
    rated_data = rated_df[["UserID", "ItemID"]].values
    n_ratings = 0
    user_train, user_test = {}, {}
    for rating_list in rated_data:
        u, ratings = rating_list
        n_ratings += len(ratings)
        user_train[u] = ratings[:-1]
        user_test[u] = ratings[-1]
    
    print(f"users size: {n_users}, item size: {n_items}, rating size: {n_ratings}")
    return user_train, user_test, n_users, n_items

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

class SeqDataset(Dataset):
    def __init__(self, user_data, n_users, n_items, args):
        self.user_data = user_data
        self.args = args
        self.n_users = n_users
        self.n_items = n_items

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, uid):
        ts = set(self.user_data[uid])

        seq = torch.zeros([self.args.maxlen], dtype=torch.int)
        pos = torch.zeros([self.args.maxlen], dtype=torch.int)
        neg = torch.zeros([self.args.maxlen], dtype=torch.int)
        nxt = self.user_data[uid][-1]
        idx = self.args.maxlen - 1

        for i in reversed(self.user_data[idx][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, self.n_items + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        uid = torch.tensor([uid])
        return uid, seq, pos, neg

def get_sparse_dense_size(model, args):
    dense_size = 0
    sparse_size = 0
    for name, param in model.named_parameters():
        if "item_emb" not in name:
            dense_size += torch.numel(param)
        else:
            sparse_size += torch.numel(param)
    return dense_size, sparse_size

if __name__ == "__main__":
    args = get_args()
    user_train, user_test, n_users, n_items = load_data(args)
    