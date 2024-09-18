from utils.parameters import get_args
from utils.util import get_emb_size, get_model_size
from data.data_util import load_data, get_rating_list, train_test_split
from data.dataset import *
from train.train_denoise import *
from models.denoise_model import *
from models.recsys_model import *

import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    args = get_args()
    item_df, user_df, rating_df = load_data(args)
    item_id_list = item_df.ItemID.unique()
    user_id_list = user_df.UserID.unique()
    ratings_dict = get_rating_list(rating_df, args)
    train_data, test_data = train_test_split(ratings_dict, args, item_id_list)
    n_user_feat = user_df.shape[-1]-1
    n_item_feat = item_df.shape[-1]-1
    n_items = len(item_id_list)
    n_users = len(user_id_list)
    # # read recsys model
    base_model = eval(args.model)(num_users=n_users, num_items=n_items, num_user_feats=n_user_feat, num_item_feats=n_item_feat, args=args)
    print("Base model size:", get_model_size(base_model))
    model_path = f"{args.root_path}/model/{args.dataset}/{args.model}/recsys_best"
    base_model.load_state_dict(torch.load(model_path, map_location=args.device))
    base_model = base_model.to(args.device)
    train_dataset = DenoiseDataset(train_data, base_model, n_users, n_items, n_user_feat, n_item_feat, args, max_item=args.max_items)
    train_loader = DataLoader(
                train_dataset, 
                batch_size=args.d_batch_size, 
                shuffle=True
                )
    # obtain the performance without noise
    test_dataset = DenoiseDataset(test_data, base_model, n_users, n_items, n_user_feat, n_item_feat, args, max_item=args.max_items, noise=False)
    test_loader = DataLoader(
                test_dataset, 
                batch_size=args.d_batch_size, 
                shuffle=False
                )
    
    mse, rmse, mae = test_model(base_model, test_loader, args, denoise=False)
    print(f"Performance without noise: RMSE: {rmse}, MSE: {mse}, MAE: {mae}")

    # obtain the performance before denoise
    test_dataset = DenoiseDataset(test_data, base_model, n_users, n_items, n_user_feat, n_item_feat, args, max_item=args.max_items)
    test_loader = DataLoader(
                test_dataset, 
                batch_size=args.d_batch_size, 
                shuffle=False
                )
    mse, rmse, mae = test_model(base_model, test_loader, args, denoise=False)
    print(f"Performance for noise without denoise: RMSE: {rmse}, MSE: {mse}, MAE: {mae}")
    # all_norms = torch.norm(base_model.embedding_user.weight, p=2, dim=-1)
    # print("average:", torch.mean(all_norms))
    # print("std:", torch.std(all_norms))
    # print("max:", torch.max(all_norms))
    # print("99%:", torch.quantile(all_norms, 0.99))
    # all_norms = torch.norm(base_model.gmf_embedding_user.weight, p=2, dim=-1)
    # print("average:", torch.mean(all_norms))
    # print("std:", torch.std(all_norms))
    # print("max:", torch.max(all_norms))
    # print("99%:", torch.quantile(all_norms, 0.99))
    # all_norms = torch.norm(base_model.ncf_embedding_user.weight, p=2, dim=-1)
    # print("average:", torch.mean(all_norms))
    # print("std:", torch.std(all_norms))
    # print("max:", torch.max(all_norms))
    # print("99%:", torch.quantile(all_norms, 0.99))
    # all_norms = torch.norm(base_model.embedding_user_feats.weight, p=2, dim=-1)
    # print("average:", torch.mean(all_norms))
    # print("std:", torch.std(all_norms))
    # print("max:", torch.max(all_norms))
    # print("99%:", torch.quantile(all_norms, 0.99))
    # initialize a denoise model
    # print("max:", base_model.user_bias.max())
    # print("99%:", torch.quantile(base_model.user_bias, 0.99))
    # print("min:", base_model.user_bias.min())
    # print("1%:", torch.quantile(base_model.user_bias, 0.01))
    emb_dim = get_emb_size(base_model, args)
    denoise_model = eval(f"{args.model}_d")(num_users=n_users, num_items=n_items, num_user_feats=n_user_feat, num_item_feats=n_item_feat, emb_dim=emb_dim, args=args)
    denoise_model = denoise_model.to(args.device)
    print("Denoise model size:", get_model_size(denoise_model))
    train_demod(user_id_list, item_id_list, train_loader, test_loader, base_model, denoise_model, args)