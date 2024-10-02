from utils.parameters import get_args
from data.data_util import load_data, get_rating_list, train_test_split
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
    ratings_dict = get_rating_list(rating_df, args)
    train_data, test_data = train_test_split(ratings_dict, args)
    # average rated items per user
    avg = []
    for u in train_data:
        avg.append(len(train_data[u]))
    avg = int(sum(avg)/len(avg))
    print("average rated item per user:", avg)

    factors_dict = {
        "MF": 64,
        "NCF": 8,
        "FM": 64,
        "DeepFM": 64,
    }
    for mod in ["MF", "NCF", "FM", "DeepFM"]:
        args.model = mod
        args.n_factors = factors_dict[mod]
        model = eval(args.model)(num_users=n_users, num_items=n_items, num_user_feats=n_user_feat, num_item_feats=n_item_feat, args=args)
        private_params = private_param_dict[args.model]
        item_params = item_param_dict[args.model]
        model_cuda = {}
        for name, param in model.named_parameters():
            if name not in private_params:
                if name in item_params:
                    model_cuda[name] = param[:avg].to(args.device)
                else:
                    model_cuda[name] = param.to(args.device)
            else:
                model_cuda[name] = param[0].to(args.device)
        
        print(f"Partial torch.cuda.memory_allocated for model {args.model}: {torch.cuda.max_memory_allocated(0)/1024/1024*2} MB")
        param_size = 0
        for name in model_cuda:
            param_size += (model_cuda[name]).numel()
        print(f"Partial param size for model {args.model}: {param_size/1000}")
        # model = model.to(args.device)
        
        del model_cuda
        torch.cuda.reset_peak_memory_stats(0)

        model_cuda = {}
        for name, param in model.named_parameters():
            if name not in private_params:
                model_cuda[name] = param.to(args.device)
            else:
                model_cuda[name] = param[0].to(args.device)
        
        param_size = 0
        for name in model_cuda:
            param_size += (model_cuda[name]).numel()
        print(f"Full param size for model {args.model}: {param_size/1000}")
        # model = model.to(args.device)
        print(f"Full torch.cuda.memory_allocated for model {args.model}: {torch.cuda.max_memory_allocated(0)/1024/1024*2} MB")
        del model_cuda
        torch.cuda.reset_peak_memory_stats(0)