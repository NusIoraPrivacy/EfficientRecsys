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
from data.dataset import ClientsSampler
import time

def test_model(model, user_id_list, item_id_list, test_dataset, args):
    prediction = []
    real_label = []
    with torch.no_grad():
        # testing
        for i in range(len(user_id_list)):
            # user_id_tensor = torch.tensor([i] * len(item_id_list)).to(args.device)
            # item_id_tensor = torch.tensor(item_id_list).to(args.device)
            try:
                this_users, this_items, true_rating, rating_vec, c_vec, item_feat, user_feat = test_dataset[i]
            except KeyError:
                continue
            if len(this_users) > 0:
                if args.model in models_w_feats:
                    user_feat = user_feat.to(args.device)
                    item_feat = item_feat.to(args.device)
                pred = model(this_users, this_items, user_feats=user_feat, item_feats=item_feat)
                # obtain rmse
                real_label.extend(true_rating.tolist())
                prediction.extend(pred.tolist())
    # print(prediction)
        mse, rmse, mae = cal_metrics(prediction, real_label, args)
    return mse, rmse, mae

def train_fl_model(user_id_list, item_id_list, train_dataset, test_dataset, model, args):
    model = model.to(args.device)
    n_items = len(item_id_list)
    n_users = len(user_id_list)
    uid_seq = DataLoader(ClientsSampler(n_users), batch_size=args.batch_size, shuffle=True)
    milestones = [args.epochs*i//10 for i in range(1, 10)]
    user_optimizers = [torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=args.lr) for i in range(n_users)]
    user_schedulers = [torch.optim.lr_scheduler.MultiStepLR(user_optimizers[i], milestones=milestones, gamma=0.5) for i in range(n_users)]
    total_rounds = args.epochs*len(uid_seq)
    milestones = [total_rounds*i//10 for i in range(1, 10)]
    server_optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=args.lr)
    server_scheduler = torch.optim.lr_scheduler.MultiStepLR(server_optimizer, milestones=milestones, gamma=0.5)

    best_rmse = 100
    best_model = copy.deepcopy(model)
    private_params = private_param_dict[args.model]
    n_rounds = 0
    patience = args.early_stop
    finish = False
    with tqdm(total=args.epochs*len(uid_seq)) as pbar:
        for epoch in range(args.epochs):
            loss_list = []
            for this_user_list in uid_seq:
                # User updates
                this_user_list = this_user_list.tolist()
                gradient_from_user = []
                loss_list = []
                item_emb_agg = 0
                public_agg = {}
                for name, param in model.named_parameters():
                    if name not in private_params:
                        public_agg[name] = 0
                n_train_users = 0
                for i in this_user_list:
                    # obtain rating prediction
                    # print("embedding before update:", model.embedding_user.weight[i])
                    try:
                        this_users, this_items, true_rating, rating_vec, c_vec, item_feat, user_feat = train_dataset[i]
                        # print("Item features:", item_feat)
                        # print("User features:", user_feat)
                        n_train_users += 1
                    except KeyError:
                        continue
                    # print(this_users, this_items, true_rating, item_feat, user_feat)
                    # update user embedding
                    if args.model == "MF" and args.als:
                        c_matrix = sp.diags(c_vec) # (n_items, n_items)
                        item_emb = model.embedding_item.weight # (n_items, emb_size)
                        Y = sp.csr_matrix(item_emb.detach().cpu().numpy()) # (n_items, emb_size)
                        p = rating_vec.T # (1， n_items)
                        YTCY = (Y.T).dot(c_matrix).dot(Y) # (emb_size, emb_size)
                        YTCp = (Y.T).dot(c_matrix).dot(p) # (emb_size, 1)
                        # this_user_embedding = spsolve((YTCY + sp.eye(args.n_factors)/n_users), YTCp) # (emb_size, 1)
                        this_user_embedding = spsolve((YTCY), YTCp)
                        this_user_embedding = (torch.tensor(this_user_embedding)).float().to(args.device)
                        with torch.no_grad():
                            model.embedding_user.weight[i] = this_user_embedding
                        predictions = model(this_users, this_items)
                        # print(predictions.shape)
                        loss = model.get_loss(predictions, true_rating)
                        loss.backward()
                        loss_list.append(loss.item())
                        # aggregate public parameters
                        for name, param in model.named_parameters():
                            if name != "embedding_user.weight":
                                public_agg[name] += param.grad
                    else:
                        if args.model in models_w_feats:
                            user_feat = user_feat.to(args.device)
                            item_feat = item_feat.to(args.device)
                        predictions = model(this_users, this_items, user_feats=user_feat, item_feats=item_feat)
                        # print(predictions.shape)
                        loss = model.get_loss(predictions, true_rating)
                        loss.backward()
                        loss_list.append(loss.item())
                        # aggregate public parameters
                        private_grads = {}
                        for name, param in model.named_parameters():
                            if name not in private_params:
                                public_agg[name] += param.grad
                            else:
                                user_emb_grad = torch.zeros(param.shape).to(args.device)
                                user_emb_grad[i] = param.grad[i]
                                private_grads[name] = user_emb_grad
                        user_optimizers[i].zero_grad()
                        for name, param in model.named_parameters():
                            if name in private_params:
                                param.grad = private_grads[name]
                        user_optimizers[i].step()
                        # user_schedulers[i].step()
                    user_optimizers[i].zero_grad()
                    # print("embedding after update:", model.embedding_user.weight[i])
                # Server update
                server_optimizer.zero_grad()
                for name, param in model.named_parameters():
                    if name not in private_params:
                        public_agg[name] = public_agg[name]/n_train_users
                        if args.regularization:
                            if "embedding_item" in name:
                                reg_term = 2 * param *  args.l2_reg_i * (public_agg[name] > 0)
                                public_agg[name] += reg_term # add regularization term
                            elif "embedding_user" in name:
                                reg_term = 2 * param *  args.l2_reg_u * (public_agg[name] > 0)
                                public_agg[name] += reg_term # add regularization term
                        param.grad = public_agg[name]
                server_optimizer.step()
                # server_scheduler.step()
                server_optimizer.zero_grad()
                torch.cuda.empty_cache()
                # for computing loss
                loss = np.mean(loss_list)
                pbar.update(1)
                pbar.set_postfix(loss=loss)
                n_rounds += 1
                loss_list.append(loss)
            mse, rmse, mae = test_model(model, user_id_list, item_id_list, test_dataset, args)
            pbar.set_postfix(loss=loss, rmse=rmse, mse=mse, mae=mae, best_rmse=best_rmse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = copy.deepcopy(model)
                patience = args.early_stop
            else:
                patience -= 1
                if patience == 0:
                    finish = True
                    break
        print("Best rmse:", best_rmse)
    save_dir = f"{args.root_path}/model/{args.dataset}/{args.model}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(best_model.state_dict(), f"{save_dir}/recsys_best_dim{int(args.n_factors)}_{args.implicit}")