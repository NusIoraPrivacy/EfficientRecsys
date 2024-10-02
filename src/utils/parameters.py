import argparse
from utils.util import str2bool, str2type
import os
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))


def get_args():
    # python

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path, 
                        help = "root path")
    parser.add_argument("--dataset", type=str, default="ml-1m", 
                        help = "dataset")
    parser.add_argument("--n_factors", type=int, default=8, 
                        help = "number of latent factors")
    parser.add_argument("--l2_reg_u", type=float, default=0.001, 
                        help = "weight on l2 regularization for user embedding")
    parser.add_argument("--l2_reg_i", type=float, default=0.001, 
                        help = "weight on l2 regularization for item embedding")
    # parser.add_argument("--alpha", type=float, default=500, 
    #                     help = "weight on non-zero ratings")
    parser.add_argument("--implicit", type=str2bool, default=False, 
                        help = "convert to implicit rating")
    parser.add_argument("--test_pct", type=float, default=0.2, 
                        help = "percentage of test data")                
    parser.add_argument("--n_train_neg", type=int, default=0, 
                        help = "number of negative samples for training")
    parser.add_argument("--n_test_neg", type=int, default=100, 
                        help = "number of negative samples for testing")
    parser.add_argument("--topk", type=int, default=10, 
                        help = "number of topk recommendations")
    parser.add_argument("--lr", type=float, default=0.05, 
                        help = "learning rate for recsys")
    parser.add_argument("--d_lr", type=float, default=0.05, 
                        help = "learning rate for denoise model")
    parser.add_argument("--batch_size", type=int, default=100, 
                        help = "number of selected client per iteration")
    parser.add_argument("--early_stop", type=int, default=20, 
                        help = "number of rounds/patience for early stop")
    parser.add_argument("--n_log_rounds", type=int, default=50, 
                        help = "number of rounds to log the accuracy")
    parser.add_argument("--epochs", type=int, default=200, 
                        help = "number of iterations")
    parser.add_argument("--d_epochs", type=int, default=50, 
                        help = "number of epochs for denoise model")
    parser.add_argument("--d_batch_size", type=int, default=100, 
                        help = "batch size for denoise model")
    parser.add_argument("--d_dim", type=int, default=5, 
                        help = "denosie dim")
    parser.add_argument("--regularization", action="store_true")
    parser.add_argument("--n_sample_items", type=int, default=100, 
                        help = "number of items to sample for each user during training")
    parser.add_argument("--max_items", type=int, default=150, 
                        help = "maximum number of sampled items per user in a denoise model")
    parser.add_argument("--model", type=str, default="MF", 
                        choices=["MF", "NCF", "FM", "DeepFM"],
                        help = "model for recommender system")
    parser.add_argument("--als", action="store_true")
    parser.add_argument("--epsilon", type=float, default=1, 
                        help = "privacy budget")
    parser.add_argument("--delta", type=float, default=1e-4, 
                        help = "privacy budget")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--debug", type=str2bool, default=False)
    args = parser.parse_args()
    return args