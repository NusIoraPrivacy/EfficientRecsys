# from Crypto.Cipher import AES
# from Crypto.Util import Counter
import sycret
import os
import time
import numpy as np
from tqdm import tqdm

def generate_ass(n):
    s = np.random.randint(0, 50000, size=(n,))
    return s

if __name__ == "__main__":
    datasets = ["ml100k", "ml1m", "ml10m", "ml25m", "yelp"]
    models = ["MF", "NCF", "FM", "DeepFM"]
    # models = ["NCF"]
    spr_dict = {"MF": [109330, 252395, 694256, 4057495, 6070090],
                "NCF": [55506, 128139, 523369, 3058727, 3828826],
                "FM": [109330, 252395, 694256, 4057495, 6070090],
                "DeepFM": [109330, 252395, 694256, 4057495, 6070090],
                }
    dense_dict = {"MF": [0, 0, 0, 0, 0],
                "NCF": [688, 688, 1512, 1512, 1060],
                "FM": [6969, 3121, 1301, 1301, 1401],
                "DeepFM": [1761065, 856370, 395798, 395798, 330002],
                }
    m_phi_list = [200, 300, 300, 500, 500]
    m_list = [1682, 3883, 10681, 62423, 93386]

    # n_users = 100
    # for i, model in enumerate(models):
    #     for j, dataset in enumerate(datasets):
    #         print(f"Model: {model}, dataset: {dataset}")
    #         n_spr = spr_dict[model][j]
    #         n_dense = dense_dict[model][j]
    #         m_phi = m_phi_list[j]
    #         m = m_list[j]
    #         eq = sycret.EqFactory(n_threads=6)
    #         keys_a, keys_b = eq.keygen(m_phi)
    #         ass = generate_ass(n_dense)
    #         keys_a_copy = keys_a.repeat(m)
    #         print(keys_a_copy.shape)
    #         t1 = time.time()
    #         # print(keys_a.shape)
    #         total_ass = 0
    #         total_vec = 0
    #         for i in range(n_users):
    #             xs = np.ones(m_phi * m)
    #             vec = eq.eval(0, xs, keys_a_copy, n_threads=6)
    #             total_vec += vec
    #             total_ass += ass
    #         t2 = time.time()
    #         print(f"Time to server aggregation for GREC: {(t2-t1)} s")
    #         ass = generate_ass(n_spr+n_dense)
    #         t1 = time.time()
    #         total_ass = 0
    #         for i in range(n_users):
    #             total_ass += ass
    #         t2 = time.time()
    #         print(f"Time to aggregation for additive secrets: {(t2-t1)} s")
            # break
    
    for i, model in enumerate(models):
        for n_users in range(100, 600, 100):
            print(f"Model: {model}, n_users: {n_users}")
            n_spr = spr_dict[model][1]
            n_dense = dense_dict[model][1]
            m_phi = m_phi_list[1]
            m = m_list[1]
            eq = sycret.EqFactory(n_threads=6)
            keys_a, keys_b = eq.keygen(m_phi)
            ass = generate_ass(n_dense)
            # keys_a_copy = keys_a.repeat(m)
            keys_a_copy = keys_a.repeat(m * n_users)
            print(keys_a_copy.shape)
            t1 = time.time()
            # print(keys_a.shape)
            total_ass = 0
            total_vec = 0
            xs = np.ones(m_phi * m * n_users)
            vec = eq.eval(0, xs, keys_a_copy, n_threads=6)
            vec = vec[:m_phi]
            for i in range(n_users):
                total_vec += vec
                total_ass += ass

            # for i in range(n_users):
            #     xs = np.ones(m_phi * m)
            #     vec = eq.eval(0, xs, keys_a_copy, n_threads=6)
            #     total_vec += vec
            #     total_ass += ass
            t2 = time.time()
            print(f"Time to server aggregation for GREC: {(t2-t1)} s")
            ass = generate_ass(n_spr+n_dense)
            t1 = time.time()
            total_ass = 0
            for i in range(n_users):
                total_ass += ass
            t2 = time.time()
            print(f"Time to aggregation for additive secrets: {(t2-t1)} s")