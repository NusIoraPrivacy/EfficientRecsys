from utils.parameters import get_args
from utils.util import get_sparse_dense_size
from data.data_util import *
from train.train_central import train_centralize_model
from models.recsys_model_colr import *
from utils.globals import *

if __name__ == "__main__":
    args = get_args()
    item_df, user_df, rating_df = load_data(args)
    if args.implicit:
        thd = pos_thd[args.dataset]
        rating_df.loc[rating_df["Rating"] < thd, "Rating"] = 0
        rating_df.loc[rating_df["Rating"] >= thd, "Rating"] = 1
    n_user_feat = user_df.shape[-1]-1
    n_item_feat = item_df.shape[-1]-1
    item_id_list = item_df.ItemID.unique()
    user_id_list = user_df.UserID.unique()
    if args.implicit:
        item_dict = get_feature_list(item_df)
        user_dict = get_feature_list(user_df)
        ratings_dict = get_rating_list(rating_df, args)
        train_data, test_data = train_test_split_neg(ratings_dict, args, item_dict, user_dict, item_id_list, train=True)
        train_data = fed2central(train_data)
    else:
        train_data, test_data = train_test_split_central(rating_df, args)
    # print(train_data[0])
    n_items = len(item_id_list)
    n_users = len(user_id_list)
    print(f"item size: {n_items}, user size: {n_users}, item fatures: {n_item_feat}, user features: {n_user_feat}")
    
    model = eval(args.model)(num_users=n_users, num_items=n_items, num_user_feats=n_user_feat, num_item_feats=n_item_feat, args=args)
    dense_size, sparse_size = get_sparse_dense_size(model, args)
    print(f"Size of Dense Vector: {dense_size}, size of sparse vector: {sparse_size}")
    train_centralize_model(n_users, n_items, n_user_feat, n_item_feat, user_id_list, train_data, test_data, model, args)