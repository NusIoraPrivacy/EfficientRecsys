from utils.parameters import get_args
from data.data_util import load_data, train_test_split_central
from torch.utils.data import DataLoader
from data.dataset import CentralDataset
from models.recsys_model import *
from tqdm import tqdm
import time
from utils.globals import *
import torch

if __name__ == "__main__":
    args = get_args()
    item_df, user_df, rating_df = load_data(args)
    n_user_feat = user_df.shape[-1]-1
    n_item_feat = item_df.shape[-1]-1
    item_id_list = item_df.ItemID.unique()
    user_id_list = user_df.UserID.unique()
    n_items = len(item_id_list)
    n_users = len(user_id_list)
    train_data, test_data = train_test_split_central(rating_df, args)
    train_dataset = CentralDataset(train_data, n_user_feat, n_item_feat, args)
    train_loader = DataLoader(train_dataset, batch_size=150, shuffle=True, pin_memory=True)

    model = eval(args.model)(num_users=n_users, num_items=n_items, num_user_feats=n_user_feat, num_item_feats=n_item_feat, args=args)
    private_params = private_param_dict[args.model]
    model_cuda = {}
    for name, param in model.named_parameters():
        if name not in private_params:
            model_cuda[name] = param.to(args.device)
    # model = model.to(args.device)
    print("torch.cuda.memory_allocated: %f MB"%(torch.cuda.max_memory_allocated(0)/1024/1024))
    # obtain inference runtime and memory cost on the whole model
    # n_test = 50
    # total_t = 0
    # for i, batch in tqdm(enumerate(train_loader, start=1)):
    #     # User updates
    #     this_users, this_items, true_rating, item_feat, user_feat = batch
    #     this_users = this_users.to(args.device)
    #     this_items = this_items.to(args.device)
    #     true_rating = true_rating.to(args.device)
    #     if args.model in models_w_feats:
    #         user_feat = user_feat.to(args.device)
    #         item_feat = item_feat.to(args.device)
    #     t1 = time.time()
    #     with torch.no_grad():
    #         predictions = model(this_users, this_items, user_feats=user_feat, item_feats=item_feat)
    #     t2 = time.time()
    #     total_t += (t2-t1)
    #     if i >= n_test:
    #         break
    # print("torch.cuda.memory_allocated: %fMB"%(torch.cuda.max_memory_allocated(0)/1024/1024))
    # print(f"Inference time per user: {total_t/n_test}")
    # obtain inference runtime and memory cost on the denoise model