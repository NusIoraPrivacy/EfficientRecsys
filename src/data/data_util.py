import pandas as pd
import numpy as np
import random
from tqdm import tqdm

def standard_id(item_df, user_df, rating_df):
    userIDs = user_df.UserID.unique()
    itemIDs = item_df.ItemID.unique()

    userid2encode = {}
    for i, userid in enumerate(userIDs):
        userid2encode[userid] = i

    itemid2encode = {}
    for i, itemid in enumerate(itemIDs):
        itemid2encode[itemid] = i

    rating_df['UserID'] = rating_df['UserID'].apply(lambda x: userid2encode[x])
    rating_df['ItemID'] = rating_df['ItemID'].apply(lambda x: itemid2encode[x])

    item_df['ItemID'] = item_df['ItemID'].apply(lambda x: itemid2encode[x])
    user_df['UserID'] = user_df['UserID'].apply(lambda x: userid2encode[x])

    return item_df, user_df, rating_df

def get_rating_list(rating_df, args, item_id_list=None):
    ratings = rating_df.values
    user_id_list = rating_df.UserID.unique()
    
    ratings_dict = {e:[] for e in user_id_list}
    counter = 0
    for record in ratings:
        ratings_dict[record[0]].append([int(record[1]), float(record[2])])
        counter += 1
    return ratings_dict

def train_test_split(ratings_dict, args, item_id_list=None):
    train_data = {}
    test_data = {}
    if item_id_list is None:
        item_id_list = rating_df.ItemID.unique()
    for user_id in ratings_dict:
        rating_list = ratings_dict[user_id]
        random.shuffle(rating_list)
        test_num = int(len(rating_list) * args.test_pct)
        train_data[user_id] = rating_list[:-test_num]
        # test_data[user_id] = {}
        # test_data[user_id]["positive"] = rating_list[-test_num:]
        test_data[user_id] = rating_list[-test_num:]

    return train_data, test_data

def load_data(args):
    data_path = f"{args.root_path}/data/{args.dataset}"
    if args.dataset == "ml-1m":
        rating_path = f"{data_path}/ratings.dat"
        rating_df = pd.read_csv(rating_path, delimiter='::', header=None, names=["UserID", "ItemID", "Rating", "Timestamp"])
        item_path = f"{data_path}/movies.dat"
        item_df = pd.read_csv(item_path, delimiter='::', header=None, names=["ItemID", "Title", "Genres"], encoding="iso-8859-1")
        # print(item_df.head())
        user_path = f"{data_path}/users.dat"
        user_df = pd.read_csv(user_path, delimiter='::', header=None, names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
        item_df, user_df, rating_df = standard_id(item_df, user_df, rating_df)
        rating_per_user = rating_df.groupby("UserID")["Rating"].count()
        # print(len(item_df))
        return item_df, user_df, rating_df
    
    if args.dataset == "ml-10m":
        rating_path = f"{data_path}/ratings.dat"
        rating_df = pd.read_csv(rating_path, delimiter='::', header=None, names=["UserID", "ItemID", "Rating", "Timestamp"])
        item_path = f"{data_path}/movies.dat"
        item_df = pd.read_csv(item_path, delimiter='::', header=None, names=["ItemID", "Title", "Genres"], encoding="iso-8859-1")
        # print(item_df.head())
        # user_path = f"{data_path}/users.dat"
        # user_df = pd.read_csv(user_path, delimiter='::', header=None, names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
        # item_df, user_df, rating_df = standard_id(item_df, user_df, rating_df)
        rating_per_user = rating_df.groupby("UserID")["Rating"].count()
        print(len(item_df))
        return item_df, rating_df
    
    if args.dataset == "ml-20m":
        rating_path = f"{data_path}/ratings.csv"
        rating_df = pd.read_csv(rating_path)
        item_path = f"{data_path}/movies.csv"
        item_df = pd.read_csv(item_path, encoding="iso-8859-1")
        # print(item_df.head())
        # user_path = f"{data_path}/users.dat"
        # user_df = pd.read_csv(user_path, delimiter='::', header=None, names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
        # item_df, user_df, rating_df = standard_id(item_df, user_df, rating_df)
        print(len(item_df))
        return item_df, rating_df
    
    if args.dataset == "ml-25m":
        rating_path = f"{data_path}/ratings.csv"
        rating_df = pd.read_csv(rating_path)
        item_path = f"{data_path}/movies.csv"
        item_df = pd.read_csv(item_path, encoding="iso-8859-1")
        # print(item_df.head())
        # user_path = f"{data_path}/users.dat"
        # user_df = pd.read_csv(user_path, delimiter='::', header=None, names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
        # item_df, user_df, rating_df = standard_id(item_df, user_df, rating_df)
        print(len(item_df))
        return item_df, rating_df
    
    if args.dataset == "bookcrossing":
        rating_path = f"{data_path}/BX-Book-Ratings.csv"
        rating_df = pd.read_csv(rating_path, delimiter=';', encoding="iso-8859-1")
        item_path = f"{data_path}/BX_Books.csv"
        item_df = pd.read_csv(item_path, delimiter=';', encoding="iso-8859-1")
        # print(item_df.head())
        # user_path = f"{data_path}/users.dat"
        # user_df = pd.read_csv(user_path, delimiter='::', header=None, names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
        # item_df, user_df, rating_df = standard_id(item_df, user_df, rating_df)
        rating_per_user = rating_df.groupby("User-ID")["Book-Rating"].count()
        print(rating_per_user.values.mean())
        print(len(item_df))
        return rating_df, item_df


# # sample unrated items
# rated_items = [i[0] for i in rating_list]
# unrated_items = [i for i in item_id_list if i not in rated_items]
# random.shuffle(unrated_items)
# sample_train_unrated = unrated_items[:args.n_train_neg]
# sample_test_unrated = unrated_items[args.n_train_neg:(args.n_train_neg+args.n_test_neg)]
# for item in sample_train_unrated:
#     train_data[user_id].append([int(item), 0])
# test_data[user_id]["negative"] = [int(item) for item in sample_test_unrated]