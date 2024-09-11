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
from utils.globals import private_param_dict

def test_model(model, user_id_list, item_id_list, test_dataset, args):
    prediction = []
    real_label = []

    # testing
    for i in range(len(user_id_list)):
        # user_id_tensor = torch.tensor([i] * len(item_id_list)).to(args.device)
        # item_id_tensor = torch.tensor(item_id_list).to(args.device)
        this_users, this_items, true_rating, rating_vec, c_vec, item_feat, user_feat = test_dataset[i]
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
    user_optimizers = [torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=args.lr) for i in range(len(user_id_list))]
    server_optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=args.lr)
    best_rmse = 100
    best_model = copy.deepcopy(model)
    private_params = private_param_dict[args.model]
    with tqdm(total=args.iters) as pbar:
        for _iter in tqdm(range(args.iters)):
            user_id_list_shuffle = user_id_list.copy()
            random.shuffle(user_id_list_shuffle)
            this_user_list = user_id_list_shuffle[:args.n_select]
            # User updates
            gradient_from_user = []
            loss_list = []
            item_emb_agg = 0
            public_agg = {}
            for name, param in model.named_parameters():
                if name not in private_params:
                    public_agg[name] = 0
            for i in this_user_list:
                # obtain rating prediction
                # print("embedding before update:", model.embedding_user.weight[i])
                this_users, this_items, true_rating, rating_vec, c_vec, item_feat, user_feat = train_dataset[i]
                # print(this_users, this_items, true_rating, item_feat, user_feat)
                # update user embedding
                if args.model == "MF" and args.als:
                    c_matrix = sp.diags(c_vec) # (n_items, n_items)
                    item_emb = model.embedding_item.weight # (n_items, emb_size)
                    Y = sp.csr_matrix(item_emb.detach().cpu().numpy()) # (n_items, emb_size)
                    p = rating_vec.T # (1， n_items)
                    YTCY = (Y.T).dot(c_matrix).dot(Y) # (emb_size, emb_size)
                    YTCp = (Y.T).dot(c_matrix).dot(p) # (emb_size, 1)
                    this_user_embedding = spsolve((YTCY + sp.eye(args.n_factors)/n_users), YTCp) # (emb_size, 1)
                    this_user_embedding = (torch.tensor(this_user_embedding)).float().to(args.device)
                    with torch.no_grad():
                        model.embedding_user.weight[i] = this_user_embedding
                    # predictions = model(this_users, this_items)
                    # # print(predictions.shape)
                    # loss = model.get_loss(predictions, true_rating)
                    # loss.backward()
                    # loss_list.append(loss.item())
                    # # aggregate public parameters
                    # for name, param in model.named_parameters():
                    #     if name != "embedding_user.weight":
                    #         public_agg[name] += param.grad
                    # calculate client's gradient pass to server
                    predictions = torch.matmul(this_user_embedding, model.embedding_item.weight.T)
                    rating_vec = torch.tensor(rating_vec.toarray(), dtype=torch.float, device=args.device)
                    errors = rating_vec - predictions
                    c_vec = torch.tensor(c_vec, dtype=torch.float, device=args.device)
                    errors = errors * c_vec
                    this_user_embedding = this_user_embedding.unsqueeze(0)
                    grad_to_server = torch.matmul(errors.T, this_user_embedding)
                    public_agg["embedding_item.weight"] += grad_to_server
                    errors = errors.squeeze()
                    errors = errors[(errors != 0)]
                    loss = torch.mean(errors ** 2)
                    loss_list.append(loss.item())
                else:
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
                user_optimizers[i].zero_grad()
                # print("embedding after update:", model.embedding_user.weight[i])

            # Server update
            server_optimizer.zero_grad()
            for name, param in model.named_parameters():
                if name not in private_params:
                    public_agg[name] = public_agg[name]/args.n_select
                    if "embedding_item" in name:
                        public_agg[name] += 2 * param / len(param) # add regularization term
                    param.grad = public_agg[name]
            server_optimizer.step()
            server_optimizer.zero_grad()

            # for computing loss
            loss = np.mean(loss_list)
            mse, rmse, mae = test_model(model, user_id_list, item_id_list, test_dataset, args)
            pbar.update(1)
            pbar.set_postfix(loss=loss, rmse=rmse, mse=mse, mae=mae)
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = copy.deepcopy(model)
    print("Best rmse:", best_rmse)
    save_dir = f"{args.root_path}/model/{args.dataset}/{args.model}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(best_model.state_dict(), f"{save_dir}/recsys_best")