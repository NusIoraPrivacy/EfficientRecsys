import torch
from torch import nn

class MF_d(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, args):
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
    
    def forward(self, ratings, item_ids, noise, init_user_emb):
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

# class NCF_d(nn.Module):
#     def __init__(self, num_users, num_items, emb_dim, args):
#         super(NCF_d, self).__init__()
#         self.embedding_item = nn.Embedding(
#             num_embeddings=num_items, embedding_dim=4)
#         self.layer1 = nn.Sequential(
#             nn.Linear(emb_dim, 4),
#             )
#         self.layer2 = nn.Sequential(
#             nn.Linear(emb_dim, 4),
#             )
#         self.p = nn.Parameter(torch.tensor([1.0]))
#         self.layer1.apply(self.init_layer)
#         self.layer2.apply(self.init_layer)
#         self.args = args
#         self.num_users = num_users
#         self.num_items = num_items
    
#     def init_layer(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out')
#             nn.init.zeros_(m.bias)

#     def forward(self, ratings, item_ids, noise, init_user_emb):
#         item_embs = self.embedding_item(item_ids) # (batch size, item size, reduce dim)
#         noise = self.layer1(noise) # (batch size, reduce dim)
#         user_emb = self.layer2(init_user_emb) # (batch size, reduce dim)
#         noise = torch.einsum("ijk, ik->ij", item_embs, noise) # (batch size, item size)
#         user_emb = torch.einsum("ijk, ik->ij", item_embs, user_emb) # (batch size, item size)
#         denoise_output = self.p * ratings + noise + user_emb # (batch size, item size)
#         return denoise_output

#     def get_loss(self, true_preds, denoise_preds, mask=None):
#         if mask is not None:
#             loss = ((true_preds - denoise_preds) ** 2) * mask
#             loss = loss.sum() / mask.sum()
#         else:
#             loss = torch.mean((true_preds - denoise_preds) ** 2)
#         reg_loss = self.embedding_item.weight.norm(2).pow(2)/self.num_items
#         loss += reg_loss
#         return loss

class NCF_d(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, args):
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

    def forward(self, ratings, item_ids, noise, init_user_emb):
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