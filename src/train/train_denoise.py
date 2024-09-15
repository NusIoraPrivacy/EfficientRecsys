import torch
from utils.globals import *
from utils.util import cal_metrics, get_noise_std, get_emb_size
import numpy as np
import heapq
from tqdm import tqdm
import os
import copy
import random

def test_model(model, test_loader, args, denoise=False, denoise_model=None):
    prediction = []
    real_label = []
    sensitivities = norm_dict[args.model]
    # testing
    with torch.no_grad():
        for batch in test_loader:
            predictions, noises, init_user_embs, item_ids, ratings, masks, item_feat = batch
            if denoise:
                predictions = denoise_model(predictions, item_ids, noises, init_user_embs, item_feat=item_feat)
            for pred, rating, mask in zip(predictions, ratings, masks):
                pred = pred[mask>0]
                rating = rating[mask>0]
                prediction.extend(pred.tolist())
                real_label.extend(rating.tolist())
        mse, rmse, mae = cal_metrics(prediction, real_label, args)
    return mse, rmse, mae

def train_demod(user_id_list, item_id_list, train_loader, test_loader, base_model, denoise_model, args):
    emb_dim = get_emb_size(base_model, args)
    total_rounds = args.d_epochs * len(train_loader)
    item_size = len(item_id_list)
    optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.d_lr)
    milestones = [total_rounds*i//5 for i in range(1, 5)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    best_rmse = 100
    best_model = copy.deepcopy(denoise_model)
    
    n_rounds = 0
    patience = args.early_stop
    finish = False
    with tqdm(total=total_rounds) as pbar:
        for epoch in range(args.d_epochs):
            for batch in train_loader:
                predictions, noises, init_user_embs, item_ids, ratings, masks, item_feat = batch
                # use the generated data to train model
                # print(item_ids.shape)
                # print(noises.shape)
                denoise_predictions = denoise_model(predictions, item_ids, noises, init_user_embs, item_feat=item_feat)
                loss = denoise_model.get_loss(denoise_predictions, ratings, masks)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # test the performance of denoise model
                pbar.update(1)
                if n_rounds % args.n_log_rounds == 0:
                    mse, rmse, mae = test_model(base_model, test_loader, args, denoise=True, denoise_model=denoise_model)
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = copy.deepcopy(denoise_model)
                        patience = args.early_stop
                    else:
                        patience -= 1
                        if patience == 0:
                            finish = True
                            break
                pbar.set_postfix(loss=loss.item(), rmse=rmse, mse=mse, mae=mae)
                n_rounds += 1
    print("Best rmse:", best_rmse)
    save_dir = f"{args.root_path}/model/{args.dataset}/{args.model}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(best_model.state_dict(), f"{save_dir}/denoise_best")