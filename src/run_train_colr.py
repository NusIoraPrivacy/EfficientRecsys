from utils.parameters import get_args
from data.data_util import load_data, get_rating_list, train_test_split
from data.dataset import ClientsDataset
from train.train_recsys_colr import train_fl_model
from models.recsys_model_colr import *

if __name__ == "__main__":
    args = get_args()
    item_df, user_df, rating_df = load_data(args)
    # print(item_df.head())
    # print(user_df.head())
    # print(rating_df.head())
    n_user_feat = user_df.shape[-1]-1
    n_item_feat = item_df.shape[-1]-1
    item_id_list = item_df.ItemID.unique()
    user_id_list = user_df.UserID.unique()
    ratings_dict = get_rating_list(rating_df, args)
    train_data, test_data = train_test_split(ratings_dict, args)
    # print(train_data[0])
    n_items = len(item_id_list)
    n_users = len(user_id_list)
    print(f"item size: {n_items}, user size: {n_users}")
    train_dataset = ClientsDataset(train_data, n_items, n_users, n_user_feat, n_item_feat, args)
    test_dataset = ClientsDataset(test_data, n_items, n_users, n_user_feat, n_item_feat, args)
    model = eval(args.model)(num_users=n_users, num_items=n_items, num_user_feats=n_user_feat, num_item_feats=n_item_feat, args=args)
    train_fl_model(user_id_list, item_id_list, train_dataset, test_dataset, model, args)