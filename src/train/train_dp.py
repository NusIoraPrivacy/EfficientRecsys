import torch
import numpy as np
from tqdm import tqdm
import os
import random
from utils.util import cal_metrics
from utils.privacy_accountant import gaussian_mechanism_dp
import copy
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from utils.globals import *
from torch.utils.data import DataLoader
from data.dataset import CentralDataset, ClientsDataset
from data.data_util import sample_item_central
from train.eval_utils import *

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

def test_model_implicit(model, test_dataset, user_id_list, args):
    hr_list = []
    ndcg_list = []
    with torch.no_grad():
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
                this_hr, this_ndcg = get_recom_metric(pred, true_rating, args)
                hr_list.extend(this_hr)
                ndcg_list.extend(this_ndcg)

    hr = sum(hr_list)/len(hr_list)
    ndcg = sum(ndcg_list)/len(ndcg_list)
    return hr, ndcg

def get_gradient_norm(model, args):
    private_params = private_param_dict[args.model]
    total_norm = 0
    for name, param in model.named_parameters():
        if (name not in private_params) and ("item_bias" not in name):
        # if (name not in private_params):
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def gaussian_noise(data_shape, norm_clip, noise_ratio, args):
    return torch.normal(0, noise_ratio * norm_clip, data_shape).to(args.device)

def avg_item_rating(train_data, n_items, args):
    max_rate = rating_range[args.dataset][1]
    avg_ratings = torch.zeros(n_items, device=args.device)
    cnt_ratings = torch.zeros(n_items, device=args.device)
    for user_rating_list in train_data:
        item, rating = int(user_rating_list[1]), user_rating_list[2]
        cnt_ratings[item] += 1
        avg_ratings[item] += rating
    # print("Total ratings: ")
    # print(avg_ratings)
    avg_ratings += gaussian_noise(avg_ratings.shape, max_rate, args.noise_ratio, args)
    # print(avg_ratings)
    # print("Total rating counts: ")
    # print(cnt_ratings)
    cnt_ratings += gaussian_noise(cnt_ratings.shape, 1, args.noise_ratio, args)
    cnt_ratings[cnt_ratings <= 1] = 1
    # print(cnt_ratings)
    avg_ratings = avg_ratings / cnt_ratings
    avg_ratings[avg_ratings < 0] = 0
    avg_ratings[avg_ratings > max_rate] = max_rate
    return avg_ratings


def train_centralize_model(n_users, n_items, n_user_feat, n_item_feat, user_id_list, train_data, test_data, model, args):
    model = model.to(args.device)
    for name, param in model.named_parameters():
        if "item_bias" in name:
            avg_ratings = avg_item_rating(train_data, n_items, args)
            param.data = avg_ratings
            # print(avg_ratings)
    n_sample_items = sample_size_dict[args.dataset]
    train_data, avg_n_per_u = sample_item_central(train_data, args, n_sample_items)
    print(avg_n_per_u)
    train_dataset = CentralDataset(train_data, n_user_feat, n_item_feat, args)
    train_batch_size = int(args.batch_size * avg_n_per_u)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True) # 50000
    if args.implicit:
        test_dataset = ClientsDataset(test_data, n_items, n_users, n_user_feat, n_item_feat, args)
        print(len(train_loader))
    else:
        test_dataset = CentralDataset(test_data, n_user_feat, n_item_feat, args)
        test_loader = DataLoader(test_dataset, batch_size=50000*5, pin_memory=True)
        print(len(train_loader), len(test_loader))
    total_rounds = args.epochs*len(train_loader)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=args.lr)
    milestones = [total_rounds*i//20 for i in range(1, 10)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    best_res = 0 if args.implicit else 100
    best_eps = 0
    save_dir = f"{args.root_path}/model/{args.dataset}/{args.model}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_rounds = 2 if args.model != "NCF" else 0
    patience = args.early_stop
    finish = False
    with tqdm(total=total_rounds) as pbar:
        for epoch in range(args.epochs):
            if epoch > 0:
                train_data, _ = sample_item_central(train_data, args, n_sample_items)
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

                clipped_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
                loss = model.get_loss_central(predictions, true_rating, dp=True)
                if args.noise_ratio == 0:
                    loss = loss.sum()
                    loss.backward(retain_graph=True)
                    for name, param in model.named_parameters():
                        if "item_bias" not in name:
                            clipped_grads[name] += param.grad
                    model.zero_grad()
                    max_norm = 0
                    train_loss = loss.item()
                else:
                    max_norm = 0
                    train_loss = 0
                    all_norms = []
                    for i in range(loss.size()[0]):
                        train_loss += loss[i].item()
                        loss[i].backward(retain_graph=True)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.norm_clip)
                        grad_norm = get_gradient_norm(model, args)
                        max_norm = max(max_norm, grad_norm)
                        all_norms.append(grad_norm)
                        for name, param in model.named_parameters():
                            clipped_grads[name] += param.grad 
                            # if i == 0:
                            #     print(name)
                            #     print(param.grad[param.grad != 0])
                        model.zero_grad()
                    # print(i)
                    # print(max_norm)
                    # print(sum(all_norms)/len(all_norms))

                private_params = private_param_dict[args.model]
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        this_grad = clipped_grads[name]
                        if name not in private_params and ("item_bias" not in name):
                            noises = gaussian_noise(this_grad.shape, max_norm, args.noise_ratio, args)
                            # print(name)
                            # print("gradients:", this_grad)
                            # print("noises:", noises)
                            this_grad += noises
                        if args.regularization:
                            if "embedding_item" in name:
                                reg_term = 2 * param *  args.l2_reg_i * (clipped_grads[name] > 0)
                                clipped_grads[name] += reg_term # add regularization term
                            elif "embedding_user" in name:
                                reg_term = 2 * param *  args.l2_reg_u * (clipped_grads[name] > 0)
                                clipped_grads[name] += reg_term # add regularization term
                        param.grad = this_grad

                loss_list.append(train_loss)
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                loss = np.mean(loss_list)
                pbar.update(1)
                pbar.set_postfix(loss=loss)
                n_rounds += 1
            if args.debug:
                break

            if args.implicit:
                hr, ndcg = test_model_implicit(model, test_dataset, user_id_list, args)
                pbar.set_postfix(loss=loss, hr=hr, ndcg=ndcg, best_hr=best_res)
                improved = (hr >= best_res)
                best_res = max(best_res, hr)
                
            else:
                mse, rmse, mae = test_model(model, test_loader, args)
                pbar.set_postfix(loss=loss, rmse=rmse, mse=mse, mae=mae, best_rmse=best_res)
                improved = (rmse <= best_res)
                best_res = min(best_res, rmse)
            if improved:
                torch.save(model.state_dict(), f"{save_dir}/recsys_best_dim{int(args.n_factors)}_{args.implicit}")
                patience = args.early_stop
                # compute privacy budget
                if args.noise_ratio != 0:
                    best_eps = gaussian_mechanism_dp(args.noise_ratio, args.delta, n_rounds)
                else:
                    best_eps = 0
                pbar.set_postfix(loss=loss, rmse=rmse, mse=mse, mae=mae, best_eps=best_eps)
            else:
                patience -= 1
                if patience == 0:
                    finish = True
                    break
        print("Best performance:", best_res)
        print("Total epsilon:", best_eps)