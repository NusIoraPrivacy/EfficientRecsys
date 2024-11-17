import tenseal as ts
import numpy as np
import time

def gencontext():
    context = ts.context(ts.SCHEME_TYPE.BFV, 8192, 1032193)
    return context

def encrypt(context, np_tensor):
    return ts.bfv_vector(context, np_tensor)

def decrypt(enc_tensor):
    return np.array(enc_tensor.decrypt())

def bootstrap(context, tensor):
    # To refresh a tensor with exhausted depth. 
    # Here, bootstrap = enc(dec())
    tmp = decrypt(tensor)
    return encrypt(context, tmp)

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

    context = gencontext()
    base_cnt = 5000
    base_vec = np.random.randint(low=0, high=500000, size=base_cnt)
    enc_base_a = encrypt(context, base_vec)
    enc_base_b = encrypt(context, base_vec)
    model = "MF"
    n_spr = spr_dict[model][1]
    n_dense = dense_dict[model][1]
    
    res_cnt = (n_spr + n_dense) % base_cnt
    res_vec = base_vec = np.random.randint(low=0, high=500000, size=res_cnt)
    enc_res_a = encrypt(context, res_vec)
    enc_res_b = encrypt(context, res_vec)
    for n_users in range(100, 600, 100):
        print(f"Model: {model}, n_users: {n_users}")
        t1 = time.time()
        for i in range(n_users):
            current_cnt = 0
            while True:
                current_cnt += base_cnt
                if current_cnt > n_spr + n_dense:
                    break
                out = enc_base_a + enc_base_b
            out = enc_res_a + enc_res_b
        t2 = time.time()
        print("Time to aggregation for BFV:", t2-t1)
    
    # a = np.random.randint(low=0, high=500000, size=8000)
    # print(a * 2)
    # context = gencontext()
    # enc_a = encrypt(context, a)
    # enc_b = encrypt(context, a)
    # print(enc_a)
    # t1 = time.time()
    # res = enc_a + enc_b
    # t2 = time.time()
    # print(decrypt(res))
    # print(t2-t1)