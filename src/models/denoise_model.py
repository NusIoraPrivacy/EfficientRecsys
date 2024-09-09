import torch
from torch import nn

class MF_d(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, args):
        super(MF_d, self).__init__()
        self.layer = nn.Sequential(
            # nn.Linear(emb_dim, 4),
            nn.Linear(emb_dim*2+num_items, 4),
            # nn.ReLU(),
            # nn.Tanh(),
            nn.LeakyReLU(),
            nn.Linear(4, num_items),
            nn.ReLU(),
            # nn.Tanh(),
            )
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.layer.apply(self.init_layer)
    
    def init_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.zeros_(m.bias)

    def forward(self, ratings, noise, init_user_emb):
        # x = self.layer(noise)
        # denoise_output = x + ratings
        _input = torch.cat([ratings, noise, init_user_emb], dim=-1)
        denoise_output = self.layer(_input)
        return denoise_output

    def get_loss(self, true_preds, denoise_preds):
        loss = torch.mean((true_preds - denoise_preds) ** 2)
        return loss

class NCF_d(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, args):
        super(NCF_d, self).__init__()
        # self.layer0 = nn.Sequential(
        #     nn.Linear(num_items, 4),
        #     nn.ReLU(),
        #     )
        # self.layer1 = nn.Sequential(
        #     nn.Linear(emb_dim, 4),
        #     nn.ReLU(),
        #     )
        # self.layer2 = nn.Sequential(
        #     nn.Linear(4, emb_dim),
        #     nn.ReLU(),
        #     )
        # self.layer3 = nn.Sequential(
        #     nn.Linear(emb_dim, 4),
        #     nn.ReLU(),
        #     )
        # self.layer4 = nn.Sequential(
        #     nn.Linear(4, num_items),
        #     nn.ReLU(),
        #     )
        self.layer = nn.Sequential(
            nn.Linear(emb_dim*2+num_items, 4),
            nn.ReLU(),
            nn.Linear(4, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 4),
            nn.ReLU(),
            nn.Linear(4, num_items),
            nn.ReLU(),
            )
        self.layer.apply(self.init_layer)
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
    
    def init_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.zeros_(m.bias)

    def forward(self, ratings, noise, init_user_emb):
        # # compute embedding
        # x = self.layer0(ratings) + self.layer1(noise)
        # x = self.layer2(x)
        # x = self.layer3(x) + self.layer0(ratings)
        # denoise_output = self.layer4(x) + ratings
        _input = torch.cat([ratings, noise, init_user_emb], dim=-1)
        denoise_output = self.layer(_input)
        return denoise_output

    def get_loss(self, true_preds, denoise_preds):
        loss = torch.mean((true_preds - denoise_preds) ** 2)
        return loss