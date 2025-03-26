import numpy as np
import torch
import math
from seqrec.models import PointWiseFeedForward

class SASRec_CoLR(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec_CoLR, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.rank = args.rank

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0).requires_grad_(False)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.A = torch.nn.Parameter(torch.zeros(self.item_num+1, args.rank))
        self.B = torch.zeros(args.rank, args.hidden_units).to(args.device)
        torch.nn.init.normal_(self.B, mean=0, std=1*math.sqrt(1/args.rank))

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def update_params(self):
        self.item_emb.weight.data = self.A @ self.B + self.item_emb.weight
        torch.nn.init.normal_(self.B, mean=0, std=1*math.sqrt(1/self.rank))
    
    def reset_A(self):
        torch.nn.init.zeros_(self.A)

    def log2feats(self, log_seqs, train=True): # TODO: fp64 and int64 as default in python, trim?
        if train:
            item_emb_mat = self.A @ self.B + self.item_emb.weight
            seqs = item_emb_mat[log_seqs]
        else:
            seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss = torch.tensor(poss, device=self.dev)
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        item_emb_mat = self.A @ self.B + self.item_emb.weight
        pos_embs = item_emb_mat[pos_seqs]
        neg_embs = item_emb_mat[neg_seqs]

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs, train=False) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(item_indices) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits