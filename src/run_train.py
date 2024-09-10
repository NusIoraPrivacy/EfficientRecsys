from utils.parameters import get_args
from data.data_util import load_data, get_rating_list, train_test_split
from data.dataset import ClientsDataset
from train.train_recsys import train_fl_model
from models.recsys_model import *

if __name__ == "__main__":
    args = get_args()
    item_df, user_df, rating_df = load_data(args)
    item_id_list = item_df.ItemID.unique()
    user_id_list = user_df.UserID.unique()
    ratings_dict = get_rating_list(rating_df, args)
    train_data, test_data = train_test_split(ratings_dict, args, item_id_list)
    print(train_data[0])
    print(test_data[0])
    n_items = len(item_id_list)
    n_users = len(user_id_list)
    train_dataset = ClientsDataset(train_data, n_items, n_users, args)
    test_dataset = ClientsDataset(test_data, n_items, n_users, args)
    model = eval(args.model)(n_users, n_items, args)
    train_fl_model(user_id_list, item_id_list, train_dataset, test_dataset, model, args)