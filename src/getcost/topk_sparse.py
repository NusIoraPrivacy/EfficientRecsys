from utils.parameters import get_args
from utils.util import get_emb_size
from data.data_util import load_data, get_rating_list, train_test_split
import torch
from torch.utils.data import DataLoader
from data.dataset import *
from models.recsys_model import *
from models.denoise_model import *
from tqdm import tqdm
import random

if __name__ == "__main__":
    args = get_args()

    item_df, user_df, rating_df = load_data(args)
    item_id_list = item_df.ItemID.unique()
    user_id_list = user_df.UserID.unique()
    ratings_dict = get_rating_list(rating_df, args)
    user_id_list = rating_df.UserID.unique()
    
    # sample 100 user in each round
    n_repeats = 1000
    avg_items = 0
    for i in tqdm(range(n_repeats)):
        all_items = []
        random.shuffle(user_id_list)
        for user_id in user_id_list[:args.batch_size]:
            rating_list = ratings_dict[user_id]
            this_items = [vec[0] for vec in rating_list]
            all_items.extend(this_items)
        all_items = set(all_items)
        avg_items += len(all_items)
    print(avg_items/n_repeats)