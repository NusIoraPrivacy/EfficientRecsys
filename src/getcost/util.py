import sycret
import os
import time
import numpy as np

def generate_ass(n):
    s = np.random.randint(0, 50000, size=(n,))
    return s

if __name__ == "__main__":
    datasets = ["ml100k", "ml1m", "ml10m", "ml25m", "yelp"]
    datasets = ["yelp"]
    # models = ["MF", "NCF", "FM", "DeepFM"]
    models = ["DeepFM"]
    method = "FSS"
    spr_dict = {"MF": {"ml100k": 109330, "ml1m": 252395, "ml10m": 694256, "ml25m": 4057495, "yelp": 6070090},
                "NCF": {"ml100k": 55506, "ml1m": 128139, "ml10m": 523369, "ml25m": 3058727, "yelp": 3828826},
                "FM": {"ml100k": 109330, "ml1m": 252395, "ml10m": 694256, "ml25m": 4057495, "yelp": 6070090},
                "DeepFM": {"ml100k": 109330, "ml1m": 252395, "ml10m": 694256, "ml25m": 4057495, "yelp": 6070090},
                }
    dense_dict = {"MF": {"ml100k": 0, "ml1m": 0, "ml10m": 0, "ml25m": 0, "yelp": 0},
                "NCF": {"ml100k":688, "ml1m": 688, "ml10m": 1512, "ml25m": 1512, "yelp": 1060},
                "FM": {"ml100k":6969, "ml1m": 3121, "ml10m": 1301, "ml25m": 1301, "yelp": 1401},
                "DeepFM": {"ml100k":1761065, "ml1m": 856370, "ml10m": 395798, "ml25m": 395798, "yelp": 330002},
                }
    m_phi_list = {"ml100k": 200, "ml1m": 300, "ml10m": 300, "ml25m": 500, "yelp": 500}

    for model in models:
        for dataset in datasets:
            all_times = []
            print(f"Model: {model}, dataset: {dataset}, method: {method}")
            for _ in range(500):
                n_spr = spr_dict[model][dataset]
                n_dense = dense_dict[model][dataset]
                m_phi = m_phi_list[dataset]
                if method == "FSS":
                    eq = sycret.EqFactory(n_threads=6)
                    t1 = time.time()
                    keys_a, keys_b = eq.keygen(m_phi)
                    ass = generate_ass(n_dense)
                    t2 = time.time()
                elif method == "ASS":
                    t1 = time.time()
                    ass = generate_ass(n_spr+n_dense)
                    t2 = time.time()
                all_times.append((t2-t1)*1000)

            _time = sum(all_times)/len(all_times)
            print(f"Time to generate secret shares: {_time} ms")