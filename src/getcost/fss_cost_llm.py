from Crypto.Cipher import AES
from Crypto.Util import Counter
import sycret
import os
import time
import numpy as np

def generate_ass(n):
    s = np.random.randint(0, 50000, size=(n,))
    return s

if __name__ == "__main__":
    models = ["RecFormer", "KAR-DeepFM", "KAR-DIN", "KAR-DIEN", "KAR-DCN"]
    # models = ["KAR-DeepFM", "KAR-DIN", "KAR-DIEN", "KAR-DCN"]
    dense_dict = {"RecFormer": 109506816,
                "KAR-DeepFM": 421715,
                "KAR-DIN": 361524,
                "KAR-DIEN": 478614,
                "KAR-DCN": 434281,
                }
    spr_dict = {"RecFormer": 2011872,
                "KAR-DeepFM": 2011872,
                "KAR-DIN": 2011872,
                "KAR-DIEN": 2011872,
                "KAR-DCN": 2011872,
                }
    
    m_phi_dict = {"RecFormer": 1000,
                "KAR-DeepFM": 200,
                "KAR-DIN": 200,
                "KAR-DIEN": 200,
                "KAR-DCN": 200,
                }

    for i, model in enumerate(models):
        print(f"Model: {model}")
        n_spr = spr_dict[model]
        n_dense = dense_dict[model]
        m_phi = m_phi_dict[model]

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