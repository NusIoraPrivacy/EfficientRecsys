from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
# from Cryptodome.Cipher import AES
# from Cryptodome.Random import get_random_bytes
from multiprocess import Pool
from tqdm import tqdm
import time
# encryption
def encrypt(m):
    return cipher.encrypt(m)
data_as_bytes = 'secret data to transmit'.encode()
# print(data_as_bytes)
key_bytes = get_random_bytes(16)
# print(key_bytes)
cipher = AES.new(key_bytes, AES.MODE_CTR, use_aesni='True')
t1 = time.time()
n_rounds = 200 * 11 * 1682
for i in range(n_rounds):
    ciphertext = cipher.encrypt(data_as_bytes)
# data_as_bytes_list = [data_as_bytes] * n_rounds
# n_process_clt = 2
# with Pool(n_process_clt) as p:
#     encrypted_data = list(tqdm(p.imap(encrypt, data_as_bytes_list)))
t2 = time.time()
print(f"time: {(t2-t1)} s")
# print(f"time: {(t2-t1)* 1000} ms")
# print(len(encrypted_data))
# print(encrypted_data[:5])
# nonce = cipher.nonce
# # decryption
# cipher = AES.new(key_bytes, AES.MODE_CTR, nonce=nonce, use_aesni='True')
# plaintext_as_bytes = cipher.decrypt(ciphertext)
# print(plaintext_as_bytes)