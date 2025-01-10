import torch
import numpy as np
from tqdm import tqdm
import os
import random
from utils.util import cal_metrics
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
            pred = model(this_users, this_items, user_feats=user_feat, item_feats=item_feat, train=False)
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

def train_centralize_model(n_users, n_items, n_user_feat, n_item_feat, user_id_list, train_data, test_data, model, args):
    model = model.to(args.device)
    train_data, avg_n_per_u = sample_item_central(train_data, args)
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
    item_params = item_param_dict[args.model]

    best_res = 0 if args.implicit else 100
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
                if args.compress == "colr":
                    model.reset_A()
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
                # ternary quantization
                if args.compress == "svd":
                    for name, param in model.named_parameters():
                        if name in item_params:
                            matrix_cpu = param.grad.cpu()
                            U_cpu, S_cpu, V_cpu = torch.svd(matrix_cpu)
                            # print(U_cpu.shape, S_cpu.shape, V_cpu.shape)
                            U_cpu, S_cpu, V_cpu = U_cpu[:, :args.rank], S_cpu[:args.rank], V_cpu[:, :args.rank]
                            U, S, V = U_cpu.to(args.device), S_cpu.to(args.device), V_cpu.to(args.device)
                            param.grad = torch.mm(torch.mm(U, torch.diag(S)), V.t())
                elif args.compress == "ternquant":
                    for name, param in model.named_parameters():
                        if name in item_params:
                            max_grad = torch.abs(param.grad).max().item()
                            probs = torch.abs(param.grad) / max_grad
                            rand_values = torch.rand(param.grad.shape, device=probs.device)
                            binary_vec = (rand_values >= probs).float()
                            ternary_grads = binary_vec * torch.sign(param.grad) * max_grad
                            param.grad = ternary_grads
                elif args.compress == "8intquant":
                    for name, param in model.named_parameters():
                        if name in item_params:
                            quant_grad = torch.quantize_per_tensor(param.grad, 0.00001, 0, torch.qint8) 
                            quant_grad = torch.dequantize(quant_grad)
                            param.grad = quant_grad
                loss_list.append(loss.item())
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                if args.compress == "colr":
                    model.update_params()
                torch.cuda.empty_cache()
                loss = np.mean(loss_list)
                pbar.update(1)
                pbar.set_postfix(loss=loss)
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
            else:
                patience -= 1
                if patience == 0:
                    finish = True
                    break
        print("Best performance:", best_res)