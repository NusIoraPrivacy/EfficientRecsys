import torch
import numpy as np
from tqdm import tqdm
import os
import heapq
import random
from utils.util import cal_metrics
import copy
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from utils.globals import private_param_dict, models_w_feats
from torch.utils.data import DataLoader
from data.dataset import CentralDataset
from data.data_util import sample_item_central

def test_model(model, test_loader, args):
    prediction = []
    real_label = []
    with torch.no_grad():
        # testing
        for batch in test_loader:
            this_users, this_items, true_rating, item_feat, user_feat = batch
            this_users = this_users.to(args.device)
            this_items = this_items.to(args.device)
            true_rating = true_rating.to(args.device)
            if args.model in models_w_feats:
                user_feat = user_feat.to(args.device)
                item_feat = item_feat.to(args.device)
            pred = model(this_users, this_items, user_feats=user_feat, item_feats=item_feat)
            # obtain rmse
            real_label.extend(true_rating.tolist())
            prediction.extend(pred.tolist())
        mse, rmse, mae = cal_metrics(prediction, real_label, args)
    return mse, rmse, mae

def train_centralize_model(n_users, n_items, n_user_feat, n_item_feat, train_data, test_data, model, args):
    model = model.to(args.device)
    train_data, avg_n_per_u = sample_item_central(train_data, args)
    print(avg_n_per_u)
    train_dataset = CentralDataset(train_data, n_user_feat, n_item_feat, args)
    test_dataset = CentralDataset(test_data, n_user_feat, n_item_feat, args)
    train_batch_size = int(args.batch_size * avg_n_per_u)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True) # 50000
    test_loader = DataLoader(test_dataset, batch_size=50000 * 5, pin_memory=True)
    print(len(train_loader), len(test_loader))
    total_rounds = args.epochs*len(train_loader)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    best_rmse = 100
    save_dir = f"{args.root_path}/model/{args.dataset}/{args.model}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_rounds = 0
    patience = args.early_stop
    finish = False
    with tqdm(total=total_rounds) as pbar:
        for epoch in range(args.epochs):
            if epoch > 0:
                train_data, _ = sample_item_central(train_data, args)
                train_dataset = CentralDataset(train_data, n_user_feat, n_item_feat, args)
                train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True) # 50000
            loss_list = []
            for batch in train_loader:
                # User updates
                this_users, this_items, true_rating, item_feat, user_feat = batch
                this_users = this_users.to(args.device)
                this_items = this_items.to(args.device)
                true_rating = true_rating.to(args.device)
                if args.model in models_w_feats:
                    user_feat = user_feat.to(args.device)
                    item_feat = item_feat.to(args.device)
                predictions = model(this_users, this_items, user_feats=user_feat, item_feats=item_feat)
                loss = model.get_loss_central(predictions, true_rating)
                loss.backward()
                loss_list.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                loss = np.mean(loss_list)
                pbar.update(1)
                pbar.set_postfix(loss=loss)

            mse, rmse, mae = test_model(model, test_loader, args)
            pbar.set_postfix(loss=loss, rmse=rmse, mse=mse, mae=mae, best_rmse=best_rmse)
            if rmse < best_rmse:
                best_rmse = rmse
                torch.save(model.state_dict(), f"{save_dir}/recsys_best")
                patience = args.early_stop
            else:
                patience -= 1
                if patience == 0:
                    finish = True
                    break
        print("Best rmse:", best_rmse)