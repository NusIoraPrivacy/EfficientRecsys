from seqrec.utils import *
from seqrec.models import SASRec
from seqrec.models_colr import SASRec_CoLR

import torch
from torch.utils.data import DataLoader

import copy
import random
import time
import numpy as np
from tqdm import tqdm

def evaluate(model, train, test, usernum, itemnum, args):

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(usernum), 10000)
    else:
        users = range(usernum)
    for u in users:

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[torch.tensor(l, device=args.device) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user

if __name__ == '__main__':
    args = get_args()
    user_train, user_test, n_users, n_items = load_data(args)
    train_dataset = SeqDataset(user_train, n_users, n_items, args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    model = SASRec(n_users, n_items, args).to(args.device)
    if args.compress == "colr":
        model = SASRec_CoLR(n_users, n_items, args).to(args.device)

    for name, param in model.named_parameters():
        if name not in ["A", "B"]:
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    model.train()
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, user_train, user_test, n_users, n_items, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    total_rounds = args.num_epochs*len(train_loader)
    n_rounds = 0
    patience = args.early_stop
    item_params = ["item_emb.weight"]
    with tqdm(total=total_rounds) as pbar:
        for epoch in range(args.num_epochs):
            if args.inference_only: break 
            loss_list = []
            for batch in train_loader: # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                if args.compress == "colr":
                    model.reset_A()
                u, seq, pos, neg = batch # tuples to ndarray
                # print(batch)
                u = u.to(args.device)
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                pos_logits, neg_logits = model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                adam_optimizer.zero_grad()
                indices = np.where(pos.cpu().detach().numpy() != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()

                if args.compress == "svd":
                    for name, param in model.named_parameters():
                        if name in item_params:
                            matrix_cpu = param.grad.cpu()
                            U_cpu, S_cpu, V_cpu = torch.svd(matrix_cpu)
                            U_cpu, S_cpu, V_cpu = U_cpu[:, :args.rank], S_cpu[:args.rank], V_cpu[:, :args.rank]
                            U, S, V = U_cpu.to(args.device), S_cpu.to(args.device), V_cpu.to(args.device)
                            param.grad = torch.mm(torch.mm(U, torch.diag(S)), V.t())

                elif args.compress == "ternquant":
                    for name, param in model.named_parameters():
                        if name in item_params:
                            max_grad = torch.abs(param.grad).max().item()
                            probs = torch.abs(param.grad) / max_grad
                            rand_values = torch.rand(param.grad.shape, device=probs.device)
                            binary_vec = (rand_values >= probs).float()
                            ternary_grads = binary_vec * torch.sign(param.grad) * max_grad
                            param.grad = ternary_grads

                elif args.compress == "8intquant":
                    for name, param in model.named_parameters():
                        if name in item_params:
                            quant_grad = torch.quantize_per_tensor(param.grad, 0.00001, 0, torch.qint8) 
                            quant_grad = torch.dequantize(quant_grad)
                            param.grad = quant_grad
                            
                adam_optimizer.step()
                if args.compress == "colr":
                    model.update_params()
                loss_list.append(loss.item())
                avg_loss = sum(loss_list)/len(loss_list)
                pbar.update(1)
                pbar.set_postfix(loss=avg_loss, best_test_ndcg=best_test_ndcg, best_test_hr=best_test_hr)
                # break
            
            if epoch % 10 == 0:
                model.eval()
                t1 = time.time() - t0
                T += t1
                t_test = evaluate(model, user_train, user_test, n_users, n_items, args)

                if t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                    best_test_ndcg = max(t_test[0], best_test_ndcg)
                    best_test_hr = max(t_test[1], best_test_hr)
                    improved = True
                else:
                    improved = False

                t0 = time.time()
                model.train()
                pbar.set_postfix(loss=avg_loss, ndcg=t_test[0], hr=t_test[1], best_test_ndcg=best_test_ndcg, best_test_hr=best_test_hr)

                if improved:
                    patience = args.early_stop
                else:
                    patience -= 1
                    if patience == 0:
                        break