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
    models = ["distillbert", "bertbase", "bertlarge", "tinyllama", "llama3.1"]
    models = ["tinyllama"]
    # models = ["NCF"]
    spr_dict = {"distillbert": 23440896, "bertbase": 23440896, "bertlarge": 31254528,
                "tinyllama": 65542144, "llama3.1": 525336576}
    dense_dict = {"distillbert": 42921984, "bertbase": 86041344, "bertlarge": 303887360,
                "tinyllama": 968976384, "llama3.1": 6979588096}
    
    m_phi = 1500

    for i, model in enumerate(models):
        print(f"Model: {model}")
        n_spr = spr_dict[model]
        n_dense = dense_dict[model]
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
        print(f"Time to generate secret keys for SecEmb: {t2-t1} s")
        t1 = time.time()
        ass = generate_ass(n_spr+n_dense)
        t2 = time.time()
        print(f"Time to generate additive secrets: {t2-t1} s")