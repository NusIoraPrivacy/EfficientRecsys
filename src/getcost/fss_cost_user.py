import sycret
import os
import time
import numpy as np

def generate_ass(n):
    s = np.random.randint(0, 100, size=(n,))
    return s

if __name__ == "__main__":
    datasets = ["ml100k", "ml1m", "ml10m", "ml25m", "yelp", "amazon"]
    # models = ["MF", "NCF", "FM", "DeepFM", "Caser", "SASRec"]
    models = ["Caser"]
    spr_dict = {"MF": [109330, 252395, 694256, 4057495, 6070090],
                "NCF": [55506, 128139, 523369, 3058727, 3828826],
                "FM": [109330, 252395, 694256, 4057495, 6070090],
                "DeepFM": [109330, 252395, 694256, 4057495, 6070090],
                "Caser": [0, 0, 0, 0, 0, int(1304465192*4/50)],
                "SASRec": [0, 0, 0, 0, 0, int(431942150*4/50)],
                }
    dense_dict = {"MF": [0, 0, 0, 0, 0],
                "NCF": [688, 688, 1512, 1512, 1060],
                "FM": [6969, 3121, 1301, 1301, 1401],
                "DeepFM": [1761065, 856370, 395798, 395798, 330002],
                "Caser": [0, 0, 0, 0, 0, 26154],
                "SASRec": [0, 0, 0, 0, 0, 41150],
                }
    
    m_phi_list = [200, 300, 300, 500, 500, 1000]

    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            if j <= 4:
                continue
            print(f"Model: {model}, dataset: {dataset}")
            n_spr = spr_dict[model][j]
            n_dense = dense_dict[model][j]
            m_phi = m_phi_list[j]
            eq = sycret.EqFactory(n_threads=6)
            # print(eq.key_len)
            # eq.key_len = 620
            t1 = time.time()
            # print(1)
            # for i in range(2):
            keys_a, keys_b = eq.keygen(m_phi)
            # print(keys_a.shape)
            # print(1)
            ass = generate_ass(n_dense)
            t2 = time.time()
            print(f"Time to generate secret keys for SecEmb: {(t2-t1)*1000} ms")
            t1 = time.time()
            ass = generate_ass(n_spr+n_dense)
            t2 = time.time()
            print(f"Time to generate additive secrets: {(t2-t1)*1000} ms")
        #     break
        # break