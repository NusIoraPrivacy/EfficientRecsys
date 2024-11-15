import heapq
import math

def get_recom_metric(predictions, ratings, args):
    pos_preds = predictions[ratings ==1].tolist()
    neg_preds = predictions[ratings == 0].tolist()
    hr_list = []
    ndcg_list = []
    for i in range(len(pos_preds)):
        pred_list = [pos_preds[i]] + neg_preds
        rank_list = [i for i, _ in heapq.nlargest(10, enumerate(pred_list), key=lambda x: x[1])]
        hr = int(0 in rank_list)
        ndcg = getNDCG(rank_list, 0)
        hr_list.append(hr)
        ndcg_list.append(ndcg)
    return hr_list, ndcg_list

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0