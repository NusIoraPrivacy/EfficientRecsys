from utils.parameters import get_args
from utils.util import get_emb_size, get_model_size
from utils.globals import *
from data.data_util import *
from data.dataset import *
from train.train_denoise import *
from models.denoise_model import *
from models.recsys_model import *

import torch
from torch.utils.data import DataLoader

def get_norm_bound(base_model, model_name, data_name):
    if model_name == "MF":
        all_norms = torch.norm(base_model.embedding_user.weight, p=2, dim=-1)
        bound1 = torch.quantile(all_norms, 0.995).item()
        bound2 = max(torch.quantile(base_model.user_bias, 0.995).item(), abs(torch.quantile(base_model.user_bias, 0.005).item()))
        return (bound1, bound2)
    
    elif model_name == "NCF":
        all_norms = torch.norm(base_model.gmf_embedding_user.weight, p=2, dim=-1)
        bound1 = torch.quantile(all_norms, 0.995).item()
        all_norms = torch.norm(base_model.ncf_embedding_user.weight, p=2, dim=-1)
        bound2 = torch.quantile(all_norms, 0.995).item()
        return (bound1, bound2)
    
    elif model_name == "FM" or model_name == "DeepFM":
        all_norms = torch.norm(base_model.embedding_user.weight, p=2, dim=-1)
        bound1 = torch.quantile(all_norms, 0.995).item()
        bound2 = 0
        if data_name in ("ml-100k", "ml-1m"):
            all_norms = torch.norm(base_model.embedding_user_feats.weight, p=2, dim=-1)
            bound2 = torch.quantile(all_norms, 0.995).item()
        bound3 = max(torch.quantile(base_model.user_bias, 0.995).item(), abs(torch.quantile(base_model.user_bias, 0.005).item()))
        return (bound1, bound2, bound3)

if __name__ == "__main__":
    args = get_args()
    item_df, user_df, rating_df = load_data(args)
    if args.implicit:
        thd = pos_thd[args.dataset]
        rating_df.loc[rating_df["Rating"] < thd, "Rating"] = 0
        rating_df.loc[rating_df["Rating"] >= thd, "Rating"] = 1
    item_id_list = item_df.ItemID.unique()
    user_id_list = user_df.UserID.unique()
    ratings_dict = get_rating_list(rating_df, args)
    if args.implicit:
        item_dict = get_feature_list(item_df)
        user_dict = get_feature_list(user_df)
        train_data, test_data = train_test_split_neg(ratings_dict, args, item_dict, user_dict, item_id_list, train=True)
    else:
        train_data, test_data = train_test_split(ratings_dict, args)
    n_user_feat = user_df.shape[-1]-1
    n_item_feat = item_df.shape[-1]-1
    n_items = len(item_id_list)
    n_users = len(user_id_list)
    # # read recsys model
    base_model = eval(args.model)(num_users=n_users, num_items=n_items, num_user_feats=n_user_feat, num_item_feats=n_item_feat, args=args)
    print("Base model size:", get_model_size(base_model))
    model_path = f"{args.root_path}/model/{args.dataset}/{args.model}/recsys_best_dim{int(args.n_factors)}_{args.implicit}"
    base_model.load_state_dict(torch.load(model_path, map_location=args.device))
    base_model = base_model.to(args.device)
    norm_bound = get_norm_bound(base_model, args.model, args.dataset)
    base_model.max_norm = norm_bound
    print(base_model.max_norm)
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
    # # initialize a denoise model
    # print("max:", base_model.user_bias.max())
    # print("99%:", torch.quantile(base_model.user_bias, 0.99))
    # print("min:", base_model.user_bias.min())
    # print("1%:", torch.quantile(base_model.user_bias, 0.01))

    # obtain the performance without noise
    train_dataset = DenoiseDataset(train_data, base_model, n_users, n_items, n_user_feat, n_item_feat, args, max_item=args.max_items)
    train_dataset.sensitivity = norm_bound
    train_loader = DataLoader(
                train_dataset, 
                batch_size=args.d_batch_size, 
                shuffle=True
                )
    test_dataset = DenoiseDataset(test_data, base_model, n_users, n_items, n_user_feat, n_item_feat, args, 
                                        # max_item=args.max_items, 
                                        noise=False, test=True)
    test_loader = DataLoader(
                test_dataset, 
                batch_size=args.d_batch_size * 5, 
                shuffle=False
                )
    if args.implicit:
        hr, ndcg = test_model_implicit(base_model, test_loader, args, denoise=False)
        print(f"Performance without noise: Hit Ratio: {hr}, NDCG: {ndcg}")
    else:
        mse, rmse, mae = test_model(base_model, test_loader, args, denoise=False)
        print(f"Performance without noise: RMSE: {rmse}, MSE: {mse}, MAE: {mae}")

    # obtain the noisy performance before denoise
    test_dataset = DenoiseDataset(test_data, base_model, n_users, n_items, n_user_feat, n_item_feat, args, max_item=args.max_items, test=True)
    test_loader = DataLoader(
                test_dataset, 
                batch_size=args.d_batch_size * 5, 
                shuffle=False
                )
    if args.implicit:
        hr, ndcg = test_model_implicit(base_model, test_loader, args, denoise=False)
        print(f"Performance without denoise: Hit Ratio: {hr}, NDCG: {ndcg}")
    else:
        mse, rmse, mae = test_model(base_model, test_loader, args, denoise=False)
        print(f"Performance without denoise: RMSE: {rmse}, MSE: {mse}, MAE: {mae}")

    emb_dim = get_emb_size(base_model, args)
    denoise_model = eval(f"{args.model}_d")(num_users=n_users, num_items=n_items, num_user_feats=n_user_feat, num_item_feats=n_item_feat, emb_dim=emb_dim, args=args)
    denoise_model = denoise_model.to(args.device)
    print("Denoise model size:", get_model_size(denoise_model))
    train_demod(user_id_list, item_id_list, train_loader, test_loader, base_model, denoise_model, args)