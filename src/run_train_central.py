from utils.parameters import get_args
from data.data_util import load_data, train_test_split_central
from data.dataset import CentralDataset
from train.train_central import train_centralize_model
from models.recsys_model import *

if __name__ == "__main__":
    args = get_args()
    item_df, user_df, rating_df = load_data(args)
    n_user_feat = user_df.shape[-1]-1
    n_item_feat = item_df.shape[-1]-1
    item_id_list = item_df.ItemID.unique()
    user_id_list = user_df.UserID.unique()
    train_data, test_data = train_test_split_central(rating_df, args)
    # print(train_data[0])
    n_items = len(item_id_list)
    n_users = len(user_id_list)
    print(f"item size: {n_items}, user size: {n_users}")
    train_dataset = CentralDataset(train_data, n_user_feat, n_item_feat, args)
    test_dataset = CentralDataset(test_data, n_user_feat, n_item_feat, args)
    model = eval(args.model)(num_users=n_users, num_items=n_items, num_user_feats=n_user_feat, num_item_feats=n_item_feat, args=args)
    train_centralize_model(n_users, n_items, train_dataset, test_dataset, model, args)