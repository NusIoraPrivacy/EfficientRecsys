from utils.parameters import get_args
from data.data_util import load_data, get_rating_list, train_test_split
from data.dataset import DenoiseDataset
from train.train_denoise import *
from models.denoise_model import *
from models.recsys_model import *

import torch

if __name__ == "__main__":
    args = get_args()
    item_df, user_df, rating_df = load_data(args)
    item_id_list = item_df.ItemID.unique()
    user_id_list = user_df.UserID.unique()
    ratings_dict = get_rating_list(rating_df, args)
    train_data, test_data = train_test_split(ratings_dict, args, item_id_list)

    # read recsys model
    base_model = eval(args.model)(len(user_id_list), len(item_id_list), args)
    model_path = f"{args.root_path}/model/{args.dataset}/{args.model}/recsys_best"
    base_model.load_state_dict(torch.load(model_path))
    base_model = base_model.to(args.device)
    n_users = len(user_id_list)
    n_items = len(item_id_list)
    train_dataset = DenoiseDataset(base_model, n_users, n_items, args)
    # obtain the performance before denoise
    mse, rmse, mae = test_model(base_model, user_id_list, item_id_list, test_data, args, denoise=False)
    print(f"RMSE: {rmse}, MSE: {mse}, MAE: {mae}")
    # all_norms = torch.norm(base_model.embedding_user.weight, p=2, dim=-1)
    # print("average:", torch.mean(all_norms))
    # print("std:", torch.std(all_norms))
    # print("max:", torch.max(all_norms))
    # print("99%:", torch.quantile(all_norms, 0.99))
    # initialize a denoise model
    emb_dim = base_model.embedding_user.weight.shape[-1]
    denoise_model = eval(f"{args.model}_d")(len(user_id_list), len(item_id_list), emb_dim, args)
    denoise_model = denoise_model.to(args.device)

    train_demod(user_id_list, item_id_list, train_dataset, test_data, base_model, denoise_model, args)