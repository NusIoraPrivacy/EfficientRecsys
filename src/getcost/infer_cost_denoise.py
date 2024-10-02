from utils.parameters import get_args
from utils.util import get_emb_size
from data.data_util import load_data, get_rating_list, train_test_split
import torch
from torch.utils.data import DataLoader
from data.dataset import *
from models.recsys_model import *
from models.denoise_model import *
from tqdm import tqdm
import time
import os

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
    save_dir = f"{args.root_path}/model/{args.dataset}/{args.model}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # read recsys model
    if args.model == "NCF":
        emb_dim = args.n_factors * 4
    else:
        emb_dim = args.n_factors
    denoise_model = eval(f"{args.model}_d")(num_users=n_users, num_items=n_items, num_user_feats=n_user_feat, num_item_feats=n_item_feat, emb_dim=emb_dim, args=args)
    denoise_model = denoise_model.to(args.device)
    print("torch.cuda.memory_allocated: %f MB"%(torch.cuda.max_memory_allocated(0)/1024/1024))

    base_model = eval(args.model)(num_users=n_users, num_items=n_items, num_user_feats=n_user_feat, num_item_feats=n_item_feat, args=args)
    base_model = base_model.to(args.device)
    print("torch.cuda.memory_allocated: %f MB"%(torch.cuda.max_memory_allocated(0)/1024/1024))

    train_dataset = DenoiseDataset(train_data, base_model, n_users, n_items, n_user_feat, n_item_feat, args, max_item=args.max_items)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    
    for batch in train_loader:
        predictions, noises, init_user_embs, item_ids, ratings, masks, item_feat = batch
        break

    n_test = 50
    total_t = 0
    denoise_model.eval()
    for i in tqdm(range(n_test)):
        t1 = time.time()
        with torch.no_grad():
            predictions = denoise_model(predictions, item_ids, noises, init_user_embs, item_feat=item_feat)
        t2 = time.time()
        total_t += (t2-t1)
    print("torch.cuda.memory_allocated: %f MB"%(torch.cuda.max_memory_allocated(0)/1024/1024))
    print(f"Inference time per user: {total_t/n_test}")