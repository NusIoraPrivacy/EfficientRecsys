import torch
from torch import nn

class MF_d(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, args, **kwargs):
        super(MF_d, self).__init__()
        self.embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=4)
        self.layer1 = nn.Sequential(
            nn.Linear(emb_dim, 4),
            # nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(emb_dim, 4),
            # nn.ReLU(),
        )
        self.p = nn.Parameter(torch.tensor([1.0]))
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.layer1.apply(self.init_layer)
        self.layer2.apply(self.init_layer)
    
    def init_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.zeros_(m.bias)
    
    def forward(self, ratings, item_ids, noise, init_user_emb, **kwargs):
        item_embs = self.embedding_item(item_ids) # (batch size, item size, reduce dim)
        noise = self.layer1(noise) # (batch size, reduce dim)
        user_emb = self.layer2(init_user_emb) # (batch size, reduce dim)
        noise = torch.einsum("ijk, ik->ij", item_embs, noise) # (batch size, item size)
        user_emb = torch.einsum("ijk, ik->ij", item_embs, user_emb) # (batch size, item size)
        denoise_output = self.p * ratings + noise + user_emb # (batch size, item size)
        return denoise_output

    def get_loss(self, true_preds, denoise_preds, mask=None):
        if mask is not None:
            loss = ((true_preds - denoise_preds) ** 2) * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = torch.mean((true_preds - denoise_preds) ** 2)
        reg_loss = self.embedding_item.weight.norm(2).pow(2)/self.num_items
        loss += reg_loss
        return loss

class NCF_d(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, args, **kwargs):
        super(NCF_d, self).__init__()
        self.gmf_embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=4)
        self.ncf_embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=4)
        self.lin_layers = nn.ModuleList([nn.Linear(emb_dim//2, 4) for i in range(4)])
        self.p = nn.Parameter(torch.tensor([1.0]))
        self.lin_layers.apply(self.init_layer)
        self.ncf_layers = nn.Sequential(
            nn.Linear(3 * 4, 2 * 4),
            nn.ReLU(),
            )
        self.final_layer = nn.Sequential(
            nn.Linear(4 * 4, 1),
            )
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
    
    def init_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.zeros_(m.bias)

    def forward(self, ratings, item_ids, noise, init_user_emb, **kwargs):
        gmf_item_embs = self.gmf_embedding_item(item_ids) # (batch size, item size, reduce dim)
        ncf_item_embs = self.ncf_embedding_item(item_ids) # (batch size, item size, reduce dim)
        init_gmf_user_emb, init_ncf_users_emb = torch.split(init_user_emb, self.emb_dim//2, dim=-1) # (batch size, reduce dim)
        gmf_noise, ncf_noise = torch.split(noise, self.emb_dim//2, dim=-1) # (batch size, reduce dim)
        init_gmf_user_emb = self.lin_layers[0](init_gmf_user_emb)
        init_ncf_users_emb = self.lin_layers[1](init_ncf_users_emb)
        gmf_noise = self.lin_layers[2](gmf_noise)
        ncf_noise = self.lin_layers[3](ncf_noise) # (batch size, reduce dim)
        gmf_noise = torch.einsum("ijk, ik->ijk", gmf_item_embs, gmf_noise) # (batch size, item size, dim)
        init_gmf_user_emb = torch.einsum("ijk, ik->ijk", gmf_item_embs, init_gmf_user_emb) # (batch size, item size, dim)
        gmf_denoise_output = torch.cat([init_gmf_user_emb, gmf_noise], dim=-1) # (batch size, item size, dim)
        init_ncf_users_emb = init_ncf_users_emb.unsqueeze(1)
        init_ncf_users_emb = init_ncf_users_emb.repeat(1, self.args.max_items, 1)
        ncf_noise = ncf_noise.unsqueeze(1)
        ncf_noise = ncf_noise.repeat(1, self.args.max_items, 1)
        ncf_input = torch.cat([init_ncf_users_emb, ncf_noise, ncf_item_embs], dim=-1)
        ncf_denoise_output = self.ncf_layers(ncf_input) # (batch size, item size, dim)
        combine_denoise_input = torch.cat([gmf_denoise_output, ncf_denoise_output], dim=-1)
        combine_denoise_output = self.final_layer(combine_denoise_input)
        combine_denoise_output = combine_denoise_output.squeeze(-1)
        denoise_output = self.p * ratings + combine_denoise_output # (batch size, item size)
        return denoise_output

    def get_loss(self, true_preds, denoise_preds, mask=None):
        if mask is not None:
            loss = ((true_preds - denoise_preds) ** 2) * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = torch.mean((true_preds - denoise_preds) ** 2)
        reg_loss = (self.gmf_embedding_item.weight.norm(2).pow(2)+self.ncf_embedding_item.weight.norm(2).pow(2))/self.num_items
        loss += reg_loss
        return loss
        
        
class FM_d(nn.Module):
    def __init__(self, num_users, num_items, num_user_feats, num_item_feats, emb_dim, args):
        super(FM_d, self).__init__()
        self.embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=6)
        self.embedding_item_feats = nn.Embedding(
            num_embeddings=num_item_feats, embedding_dim=6)
        self.lin_layers = nn.ModuleList([nn.Linear(emb_dim, 6) for i in range(4)])
        # self.embed_output_dim = (3 + 2*num_user_feats + num_item_feats) * 4
        # self.hidden_layers = nn.Sequential(
        #     nn.Linear(self.embed_output_dim, 2 * 4),
        #     nn.ReLU(),
        #     nn.Linear(2 * 4, 1),
        #     )
        self.p1 = nn.Parameter(torch.tensor([1.0]))
        self.p2 = nn.Parameter(torch.tensor([0.0]))
        self.p3 = nn.Parameter(torch.tensor([0.0]))
        # self.p4 = nn.Parameter(torch.tensor([0.0]))
        # self.p5 = nn.Parameter(torch.tensor([0.0]))
        nn.init.kaiming_normal_(self.embedding_item.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_item_feats.weight, mode='fan_out')
        self.lin_layers.apply(self.init_layer)
        # self.hidden_layers.apply(self.init_layer)
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.num_user_feats = num_user_feats
        self.num_item_feats = num_item_feats
        self.emb_dim = emb_dim
    
    def init_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.zeros_(m.bias)
            
    def forward(self, ratings, item_ids, noise, init_user_emb, item_feat):
        item_size = item_ids.shape[-1]
        # transform raw user embedding and noises
        init_user_embs, init_user_feat_embs = torch.split(init_user_emb, [1, self.num_user_feats], dim=1) # batch size, num_feat, dim
        user_emb_noises, user_feat_emb_noises = torch.split(noise, [1, self.num_user_feats], dim=1) # batch size, num_feat, dim
        init_user_embs = self.lin_layers[0](init_user_embs)
        init_user_feat_embs = self.lin_layers[1](init_user_feat_embs)
        user_emb_noises = self.lin_layers[2](user_emb_noises)
        user_feat_emb_noises = self.lin_layers[3](user_feat_emb_noises)
        all_user_embs = torch.cat([init_user_embs, init_user_feat_embs], dim=1)  # batch size, num_feat, dim
        all_noises = torch.cat([user_emb_noises, user_feat_emb_noises], dim=1)  # batch size, num_feat, dim
        all_user_embs = all_user_embs.unsqueeze(1).repeat(1, item_size, 1, 1)  # batch size, item size num_feat, dim
        all_noises = all_noises.unsqueeze(1).repeat(1, item_size, 1, 1) # batch size, item size, num_feat, dim
        # obtain item embs
        item_ids = item_ids.unsqueeze(-1)
        item_embs = self.embedding_item(item_ids) # batch size, item size, 1, dim
        item_feat_embs = torch.einsum("ij, bmi->bmij", self.embedding_item_feats.weight, item_feat) # batch size, item size, num_feat, dim
        all_item_embs = torch.cat([item_embs, item_feat_embs], dim=2)
        all_embs = torch.cat([all_user_embs, all_item_embs], dim=2)
        pos_terms = all_embs.sum(2).pow(2)
        neg_terms = all_embs.pow(2).sum(2)
        denoise_output = (pos_terms - neg_terms).sum(-1)*0.5
        all_noisy_embs = torch.cat([all_user_embs+all_noises, all_item_embs], dim=2)
        pos_terms = all_noisy_embs.sum(2).pow(2)
        neg_terms = all_noisy_embs.pow(2).sum(2)
        denoise_output = self.p1 * denoise_output + self.p2 * ratings + self.p3 * (pos_terms-neg_terms).sum(-1)*0.5
        # user_item = torch.cat([all_user_embs, all_item_embs], dim=2)
        # user_item = (user_item.sum(2).pow(2) - user_item.pow(2).sum(2)).sum(-1)*0.5
        # user_noise = torch.cat([all_user_embs, all_noises], dim=2)
        # user_noise = (user_noise.sum(2).pow(2) - user_noise.pow(2).sum(2)).sum(-1)*0.5
        # item_noise = torch.cat([all_item_embs, all_noises], dim=2)
        # item_noise = (item_noise.sum(2).pow(2) - item_noise.pow(2).sum(2)).sum(-1)*0.5
        # all_inputs = torch.cat([all_user_embs, all_item_embs, all_noises], dim=2) # batch size, item size, num_feat, dim
        # bz, item_size, n_feats, n_dim = all_inputs.shape
        # all_inputs = all_inputs.view(bz, item_size, n_feats * n_dim)
        # denoise_output = self.hidden_layers(all_inputs)
        # denoise_output = denoise_output.squeeze(-1)
        # denoise_output = self.p1 * ratings + self.p2 * denoise_output + self.p3 * user_item + self.p4 * user_noise + self.p5 * item_noise
        # denoise_output = self.p1 * item_noise + self.p2 * user_noise + self.p3 * user_item + self.p4 * ratings
        return denoise_output

    def get_loss(self, true_preds, denoise_preds, mask=None):
        if mask is not None:
            loss = ((true_preds - denoise_preds) ** 2) * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = torch.mean((true_preds - denoise_preds) ** 2)
        reg_loss = (self.embedding_item.weight.norm(2).pow(2)+self.embedding_item_feats.weight.norm(2).pow(2))/self.num_items
        loss += reg_loss
        return loss

class DeepFM_d(nn.Module):
    def __init__(self, num_users, num_items, num_user_feats, num_item_feats, emb_dim, args):
        super(FM_d, self).__init__()
        self.embedding_item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=4)
        self.embedding_item_feats = nn.Embedding(
            num_embeddings=num_item_feats, embedding_dim=4)
        self.lin_layers = nn.ModuleList([nn.Linear(emb_dim, 4) for i in range(4)])
        self.embed_output_dim = (3 + 2*num_user_feats + num_item_feats) * 4
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.embed_output_dim, 2 * 4),
            nn.BatchNorm1d(self.args.max_items),
            nn.ReLU(),
            nn.Linear(self.embed_output_dim, 2 * 4),
            nn.BatchNorm1d(self.args.max_items),
            nn.ReLU(),
            nn.Linear(2 * 4, 1),
            )
        self.p1 = nn.Parameter(torch.tensor([1.0]))
        self.p2 = nn.Parameter(torch.tensor([0.0]))
        self.p3 = nn.Parameter(torch.tensor([0.0]))
        self.p4 = nn.Parameter(torch.tensor([0.0]))
        self.p5 = nn.Parameter(torch.tensor([0.0]))
        nn.init.kaiming_normal_(self.embedding_item.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.embedding_item_feats.weight, mode='fan_out')
        self.lin_layers.apply(self.init_layer)
        self.hidden_layers.apply(self.init_layer)
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.num_user_feats = num_user_feats
        self.num_item_feats = num_item_feats
        self.emb_dim = emb_dim
    
    def init_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.zeros_(m.bias)
            
    def forward(self, ratings, item_ids, noise, init_user_emb, item_feat):
        item_size = item_ids.shape[-1]
        # transform raw user embedding and noises
        init_user_embs, init_user_feat_embs = torch.split(init_user_emb, [1, self.num_user_feats], dim=1) # batch size, num_feat, dim
        user_emb_noises, user_feat_emb_noises = torch.split(noise, [1, self.num_user_feats], dim=1) # batch size, num_feat, dim
        init_user_embs = self.lin_layers[0](init_user_embs)
        init_user_feat_embs = self.lin_layers[1](init_user_feat_embs)
        user_emb_noises = self.lin_layers[2](user_emb_noises)
        user_feat_emb_noises = self.lin_layers[3](user_feat_emb_noises)
        all_user_embs = torch.cat([init_user_embs, init_user_feat_embs], dim=1)  # batch size, num_feat, dim
        all_noises = torch.cat([user_emb_noises, user_feat_emb_noises], dim=1)  # batch size, num_feat, dim
        all_user_embs = all_user_embs.unsqueeze(1).repeat(1, item_size, 1, 1)  # batch size, item size num_feat, dim
        all_noises = all_noises.unsqueeze(1).repeat(1, item_size, 1, 1) # batch size, item size, num_feat, dim
        # obtain item embs
        item_ids = item_ids.unsqueeze(-1)
        item_embs = self.embedding_item(item_ids) # batch size, item size, 1, dim
        item_feat_embs = torch.einsum("ij, bmi->bmij", self.embedding_item_feats.weight, item_feat) # batch size, item size, num_feat, dim
        all_item_embs = torch.cat([item_embs, item_feat_embs], dim=2)
        all_embs = torch.cat([all_user_embs, all_item_embs], dim=2)
        pos_terms = all_embs.sum(2).pow(2)
        neg_terms = all_embs.pow(2).sum(2)
        fm_denoise_output = (pos_terms - neg_terms).sum(-1)*0.5
        all_noisy_embs = torch.cat([all_user_embs+all_noises, all_item_embs], dim=2)
        pos_terms = all_noisy_embs.sum(2).pow(2)
        neg_terms = all_noisy_embs.pow(2).sum(2)
        all_inputs = torch.cat([all_embs, all_noises], dim=2)
        bz, item_size, n_feats, n_dim = all_inputs.shape
        all_inputs = all_inputs.view(bz, item_size, n_feats * n_dim)
        lin_denoise_output = self.hidden_layers(all_inputs)
        lin_denoise_output = lin_denoise_output.squeeze(-1)
        denoise_output = self.p1 * fm_denoise_output + self.p2 * lin_denoise_output + self.p3 * ratings
        return denoise_output

    def get_loss(self, true_preds, denoise_preds, mask=None):
        if mask is not None:
            loss = ((true_preds - denoise_preds) ** 2) * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = torch.mean((true_preds - denoise_preds) ** 2)
        reg_loss = (self.embedding_item.weight.norm(2).pow(2)+self.embedding_item_feats.weight.norm(2).pow(2))/self.num_items
        loss += reg_loss
        return loss