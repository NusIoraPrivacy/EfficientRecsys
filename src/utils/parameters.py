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
    parser.add_argument("--l2_reg", type=float, default=2, 
                        help = "weight on l2 regularization")
    # parser.add_argument("--alpha", type=float, default=500, 
    #                     help = "weight on non-zero ratings")
    parser.add_argument("--implicit", type=str2bool, default=False, 
                        help = "convert to implicit rating")
    parser.add_argument("--test_pct", type=float, default=0.2, 
                        help = "percentage of test data")                
    parser.add_argument("--n_train_neg", type=int, default=5, 
                        help = "number of negative samples for training")
    parser.add_argument("--n_test_neg", type=int, default=100, 
                        help = "number of negative samples for testing")
    parser.add_argument("--topk", type=int, default=10, 
                        help = "number of topk recommendations")
    parser.add_argument("--lr", type=float, default=0.05, 
                        help = "learning rate for recsys")
    parser.add_argument("--d_lr", type=float, default=0.01, 
                        help = "learning rate for denoise model")
    parser.add_argument("--n_select", type=int, default=600, 
                        help = "number of selected client per iteration")
    parser.add_argument("--iters", type=int, default=2000, 
                        help = "number of iterations")
    parser.add_argument("--d_epochs", type=int, default=1000, 
                        help = "number of epochs for denoise model")
    parser.add_argument("--d_batch_size", type=int, default=100, 
                        help = "batch size for denoise model")
    parser.add_argument("--model", type=str, default="MF", 
                        choices=["MF", "NCF"],
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