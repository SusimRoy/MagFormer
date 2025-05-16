import copy
import json
import math
import re
import collections

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import Optional
import random
from .rmsnorm import RMSNorm

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)

ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}

class SwiGLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension,dimension)
        self.linear_2 = nn.Linear(dimension,dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear_2(x)

        return swiglu

class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        assert n_state % cfg.MODEL.N_HEAD == 0
        self.n_head = cfg.MODEL.N_HEAD
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        # self.c_proj = Conv1D(n_state, 1, nx)
        self.c_proj = nn.Linear(nx, nx, bias=False)

        self.resid_dropout = nn.Dropout(cfg.MODEL.RESID_PDROP)

        self.head_dim = nx // cfg.MODEL.N_HEAD // 2
        self.scaling = self.head_dim ** -0.5
        self.num_heads = cfg.MODEL.N_HEAD

        # self.m1 = nn.Parameter(torch.empty(2, 2*cfg.MODEL.N_HEAD, 1, self.head_dim))
        self.m1 = nn.Parameter(torch.empty(2*cfg.MODEL.N_HEAD, cfg.MODEL.LANDMARK, 1))
        # self.m2 = nn.Parameter(torch.empty(2, 2*cfg.MODEL.N_HEAD, self.head_dim, 1))
        self.m2 = nn.Parameter(torch.empty(2*cfg.MODEL.N_HEAD, 1, cfg.MODEL.LANDMARK))
        
        # Apply Kaiming initialization
        nn.init.kaiming_normal_(self.m1, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.m2, mode='fan_out', nonlinearity='relu')
        self.q_proj = nn.Linear(nx, nx, bias=False)
        self.k_proj = nn.Linear(nx, nx, bias=False)
        self.v_proj = nn.Linear(nx, nx, bias=False)

        # self.lambda_init = self.lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)


    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3*depth)

    def _attn(self, l, q, k, v, num_landmark, rns_indices):
        self.lambda_init = self.lambda_init_fn(l)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).transpose(-1,-2)
        v = v.transpose(1, 2)
        data_length = q.shape[2]
        landmark = torch.Tensor(random.sample(range(data_length),num_landmark)).long()
        sq = q[:,:,landmark,:].contiguous()
        sk = k[:,:,:,landmark].contiguous()

        sq  = sq / torch.linalg.norm(sq, ord=2, dim=3, keepdim=True)
        sk = sk / torch.linalg.norm(sk, ord=2, dim=2, keepdim=True)

        sq = (self.m1.unsqueeze(0) * sq).contiguous()
        sk = (self.m2.unsqueeze(0) * sk).contiguous()

        w1 = torch.matmul(q, sk)
        w2 = torch.matmul(sq, k)
        # print("w1: ", w1.shape)
        # print("w2: ", w2.shape)
        w = torch.matmul(w1, w2)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        if self.scale:
            w = self.scaling * w
        return self.rns(l, w, v, rns_indices, lambda_full)
    
    def visualize_threshold_attention(self, w, name, l, threshold=0.5, sample_idx=0, head_idx=1):
        """
        Visualize attention distribution with threshold highlighting using histograms
        Args:
            w: attention weights tensor
            name: name for the plot
            threshold: threshold value to split distribution (default 0.5)
            sample_idx: which batch sample to visualize
            head_idx: which attention head to visualize
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Detach from computation graph and move to CPU
        if len(w.shape) == 5:  # [bs, heads, 2, seq_len, seq_len]
            w_vis = w[sample_idx, head_idx].detach().cpu().numpy()
        elif len(w.shape) == 4:  # [bs, heads, seq_len, seq_len]
            w_vis = w[sample_idx, head_idx].detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected shape for attention matrix: {w.shape}")
        
        # Flatten the attention matrix to analyze distribution
        w_flat = w_vis.flatten()
        
        # Create separate arrays for values above and below threshold
        high_values = w_flat[w_flat > threshold]
        low_values = w_flat[w_flat <= threshold]
        
        # Create the histograms plot
        plt.figure(figsize=(18, 8))
        
        # First subplot - histogram of values > threshold
        plt.subplot(1, 2, 1)
        if len(high_values) > 0:
            plt.hist(high_values, bins=30, color='red', alpha=0.7)
            plt.axvline(x=threshold, color='black', linestyle='--', 
                    label=f'Threshold: {threshold}')
            plt.title(f"Attention values > {threshold} (count: {len(high_values)})")
            plt.xlabel("Attention value")
            plt.ylabel("Frequency")
            plt.legend()
        else:
            plt.text(0.5, 0.5, f"No values > {threshold}", 
                    ha='center', va='center', fontsize=14)
        
        # Second subplot - histogram of values <= threshold
        plt.subplot(1, 2, 2)
        if len(low_values) > 0:
            plt.hist(low_values, bins=30, color='blue', alpha=0.7)
            plt.axvline(x=threshold, color='black', linestyle='--',
                    label=f'Threshold: {threshold}')
            plt.title(f"Attention values <= {threshold} (count: {len(low_values)})")
            plt.xlabel("Attention value")
            plt.ylabel("Frequency")
            plt.legend()
        else:
            plt.text(0.5, 0.5, f"No values <= {threshold}", 
                    ha='center', va='center', fontsize=14)
        
        # Additional statistics
        plt.figtext(0.5, 0.01, 
                f"Statistics: Mean={w_flat.mean():.4f}, Median={np.median(w_flat):.4f}, "
                f"Min={w_flat.min():.4f}, Max={w_flat.max():.4f}", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
#         # Save the plot
        save_path = f"/home/csgrad/susimmuk/CSE676/NFormer/vizz/attention_hist_{name}_sample{sample_idx}_head{head_idx}_at_layer_{l}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
#         # print(f"Saved histogram visualization to {save_path}")

    def rns(self, l, w, v, rns_indices, lambda_full):
        # Exp 1
        bs,hn,dl,_ = w.shape
        rns_indices = rns_indices.unsqueeze(1).repeat(1,hn,1,1)
        mask = torch.zeros_like(w).scatter_(3, rns_indices,torch.ones_like(rns_indices, dtype=w.dtype))
        mask = mask * mask.transpose(2,3)
        if 'cuda' in str(w.device):
            mask = mask.cuda()
        else:
            mask = mask.cpu()
        w = w * mask + -1e9 * (1 - mask)
        w = F.softmax(w,dim=3)
        # self.visualize_threshold_attention(w, "rns_and_diff", l, threshold=0.5, sample_idx=0, head_idx=2)
        w = w.view(bs, self.num_heads, 2, dl, dl)
        w1 = (w[:, :, 0] - lambda_full*w[:, :, 1])
        w2 = ((1-lambda_full)*w[:, 1, :] - w[:, 0, :])
        w = w1+w2
        # w = torch.clip(w,min=0)
        # self.visualize_threshold_attention(w, "rns_afterdifftransformer", l, threshold=0.3, sample_idx=0, head_idx=0)
        a_v = torch.matmul(w, v)
        #Exp 2
        # bs,hn,dl,_ = w.shape
        # rns_indices = rns_indices.unsqueeze(1).repeat(1,hn,1,1)
        # mask = torch.zeros_like(w).scatter_(3, rns_indices,torch.ones_like(rns_indices, dtype=w.dtype))
        # mask = mask * mask.transpose(2,3)
        # if 'cuda' in str(w.device):
        #     mask = mask.cuda()
        # else:
        #     mask = mask.cpu()
        # w = w * mask + -1e9 * (1 - mask)
        # w = F.softmax(w,dim=3)
        # a_v = torch.matmul(w, v)
        # Exp 3
        # bs,hn,dl,_ = w.shape
        # attn_weights2 = F.softmax(w,dim=3)
        # attn_weights2 = attn_weights2.view(bs, self.num_heads, 2, dl, dl)
        # attn_weights2 = attn_weights2[:, :, 0] - lambda_full * attn_weights2[:, :, 1]
        # attn_avg = attn_weights2
        # a_v = torch.matmul(attn_avg, v)   
        # Exp 4
        # bs,hn,dl,_ = w.shape
        # rns_indices = rns_indices.unsqueeze(1).repeat(1,hn,1,1)
        # mask = torch.zeros_like(w).scatter_(3, rns_indices,torch.ones_like(rns_indices, dtype=w.dtype))
        # mask = mask * mask.transpose(2,3)
        # if 'cuda' in str(w.device):
        #     mask = mask.cuda()
        # else:
        #     mask = mask.cpu()
        # # print("mask: ", mask.shape)
        # # print("w: ", w.shape)
        # attn_weights1 = w * mask + -1e9 * (1 - mask)
        # attn_weights1 = F.softmax(attn_weights1,dim=3)
        # attn_weights1 = attn_weights1.view(bs, self.num_heads, 2, dl, dl)
        # attn_weights1_avg = 0.5*(attn_weights1[:, :, 0] + attn_weights1[:, :, 1])

        # attn_weights2 = F.softmax(w,dim=3)
        # attn_weights2 = attn_weights2.view(bs, self.num_heads, 2, dl, dl)
        # attn_weights2 = attn_weights2[:, :, 0] - lambda_full * attn_weights2[:, :, 1]
        # attn_avg = 0.5*(attn_weights1_avg + attn_weights2)
        # a_v = torch.matmul(attn_avg, v)
        return a_v


    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, l, x, num_landmark, rns_indices):
        bsz, tgt_len, embed_dim = x.size()
        # print("x: ", x.shape)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        query = q.view(bsz, tgt_len, 2 * self.n_head, self.head_dim)
        key = k.view(bsz, tgt_len, 2 * self.n_head, self.head_dim)
        value = v.view(bsz, tgt_len, self.n_head, 2 * self.head_dim)
        a = self._attn(l, query, key, value, num_landmark, rns_indices)
        # a = self.merge_heads(a)
        attn = self.subln(a)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        attn = self.c_proj(attn)
        # attn = self.resid_dropout(attn)
        return attn


class MLP(nn.Module):
    def __init__(self, n_state, cfg):
        super(MLP, self).__init__()
        nx = cfg.MODEL.N_EMBD
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[cfg.MODEL.AFN]
        self.dropout = nn.Dropout(cfg.MODEL.RESID_PDROP)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg.MODEL.N_EMBD
        self.split_size = nx
        self.n_head = cfg.MODEL.N_HEAD
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)
        self.c_attn = Conv1D(nx * 3, 1, nx)
        self.resid_dropout = nn.Dropout(cfg.MODEL.RESID_PDROP)
        self.c_proj = Conv1D(nx, 1, nx)
        self.swglu1 = SwiGLU(nx)
        self.rms_norm1 = RMSNorm(nx, eps=1e-5, elementwise_affine=True)
        self.rms_norm2 = RMSNorm(nx, eps=1e-5, elementwise_affine=True)

    def forward(self, l, x, num_landmark, rns_indices):
        a = self.attn(l, x, num_landmark, rns_indices)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        x_l = self.ln_2(n + m)
        return x_l


class NFormer(nn.Module):
    """ NFormer model """

    def __init__(self, cfg, vocab=40990, n_ctx=1024, num_classes = 751):
        super(NFormer, self).__init__()
        # if self.training:
        #     self.num_classes = 767
        # else:
        #     self.num_classes = 700
        self.num_classes = num_classes
        
        block = Block(n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.MODEL.N_LAYER)])

        self.bottleneck = nn.BatchNorm1d(cfg.MODEL.N_EMBD)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        
        self.classifier = nn.Linear(cfg.MODEL.N_EMBD, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.topk = cfg.MODEL.TOPK
        self.num_landmark = cfg.MODEL.LANDMARK

    def forward(self, x):
        _, rns_indices = torch.topk(torch.bmm(x/torch.norm(x,p=2,dim=2,keepdim=True),(x/torch.norm(x,p=2,dim=2,keepdim=True)).transpose(1,2)), self.topk, dim=2)
        for l, block in enumerate(self.h):
            x = block(l, x, self.num_landmark, rns_indices)

        bs,dl,d = x.shape
        x = x.reshape(bs*dl,d)
        feat = self.bottleneck(x)
        cls_score = self.classifier(feat)
        x = x.reshape(bs,dl,d)
        feat = feat.reshape(bs,dl,d)
        cls_score = cls_score.reshape(bs,dl,-1)

        if self.training:
            return cls_score, x
        else:
            return feat

# import copy
# import json
# import math
# import re
# import collections

# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# import random

# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
#         nn.init.constant_(m.bias, 0.0)
#     elif classname.find('Conv') != -1:
#         nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0.0)
#     elif classname.find('BatchNorm') != -1:
#         if m.affine:
#             nn.init.constant_(m.weight, 1.0)
#             nn.init.constant_(m.bias, 0.0)

# def weights_init_classifier(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         nn.init.normal_(m.weight, std=0.001)
#         if m.bias:
#             nn.init.constant_(m.bias, 0.0)

# def gelu(x):
#     return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# def swish(x):
#     return x * torch.sigmoid(x)

# ACT_FNS = {
#     'relu': nn.ReLU,
#     'swish': swish,
#     'gelu': gelu
# }


# class LayerNorm(nn.Module):
#     "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

#     def __init__(self, n_state, e=1e-5):
#         super(LayerNorm, self).__init__()
#         self.g = nn.Parameter(torch.ones(n_state))
#         self.b = nn.Parameter(torch.zeros(n_state))
#         self.e = e

#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.e)
#         return self.g * x + self.b


# class Conv1D(nn.Module):
#     def __init__(self, nf, rf, nx):
#         super(Conv1D, self).__init__()
#         self.rf = rf
#         self.nf = nf
#         if rf == 1:  # faster 1x1 conv
#             w = torch.empty(nx, nf)
#             nn.init.normal_(w, std=0.02)
#             self.w = Parameter(w)
#             self.b = Parameter(torch.zeros(nf))
#         else:  # was used to train LM
#             raise NotImplementedError

#     def forward(self, x):
#         if self.rf == 1:
#             size_out = x.size()[:-1] + (self.nf,)
#             x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
#             x = x.view(*size_out)
#         else:
#             raise NotImplementedError
#         return x


# class Attention(nn.Module):
#     def __init__(self, nx, n_ctx, cfg, scale=False):
#         super(Attention, self).__init__()
#         n_state = nx
#         assert n_state % cfg.MODEL.N_HEAD == 0
#         self.n_head = cfg.MODEL.N_HEAD
#         self.split_size = n_state
#         self.scale = scale
#         self.c_attn = Conv1D(n_state * 3, 1, nx)
#         self.c_proj = Conv1D(n_state, 1, nx)

#         self.resid_dropout = nn.Dropout(cfg.MODEL.RESID_PDROP)

#         # self.m1 = nn.Parameter(torch.empty(2, 2*cfg.MODEL.N_HEAD, 1, self.head_dim))
#         # self.m1 = nn.Parameter(torch.empty(cfg.MODEL.N_HEAD, cfg.MODEL.LANDMARK, 1))
#         # # self.m2 = nn.Parameter(torch.empty(2, 2*cfg.MODEL.N_HEAD, self.head_dim, 1))
#         # self.m2 = nn.Parameter(torch.empty(cfg.MODEL.N_HEAD, 1, cfg.MODEL.LANDMARK))
        
#         # # Apply Kaiming initialization
#         # nn.init.kaiming_normal_(self.m1, mode='fan_out', nonlinearity='relu')
#         # nn.init.kaiming_normal_(self.m2, mode='fan_out', nonlinearity='relu')

#     def _attn(self,l,  q, k, v, num_landmark, rns_indices):
#         data_length = q.shape[2]
#         landmark = torch.Tensor(random.sample(range(data_length),num_landmark)).long()
        
#         sq = q[:,:,landmark,:].contiguous()
#         sk = k[:,:,:,landmark].contiguous()

#         # sq  = sq / torch.linalg.norm(sq, ord=2, dim=3, keepdim=True)
#         # sk = sk / torch.linalg.norm(sk, ord=2, dim=2, keepdim=True)

#         # sq = (self.m1.unsqueeze(0) * sq).contiguous()
#         # sk = (self.m2.unsqueeze(0) * sk).contiguous()

#         w1 = torch.matmul(q, sk)
#         w2 = torch.matmul(sq, k)
#         w = torch.matmul(w1, w2)

#         if self.scale:
#             w = w / math.sqrt(v.size(-1))
#         return self.rns(l, w, v, rns_indices)
    
    
#     def visualize_threshold_attention(self, w, name, l, threshold=None, sample_idx=0, head_idx=0):
#         """
#         Visualize attention distribution with appropriate scaling for very small values
        
#         Args:
#             w: attention weights tensor
#             name: name for the plot
#             l: layer number
#             threshold: threshold value (if None, will be set automatically)
#             sample_idx: which batch sample to visualize
#             head_idx: which attention head to visualize
#         """
#         import matplotlib.pyplot as plt
#         import numpy as np
        
#         # Detach from computation graph and move to CPU
#         if len(w.shape) == 5:  # [bs, heads, 2, seq_len, seq_len]
#             w_vis = w[sample_idx, head_idx].detach().cpu().numpy()
#         elif len(w.shape) == 4:  # [bs, heads, seq_len, seq_len]
#             w_vis = w[sample_idx, head_idx].detach().cpu().numpy()
#         else:
#             raise ValueError(f"Unexpected shape for attention matrix: {w.shape}")
        
#         # Flatten the attention matrix to analyze distribution
#         w_flat = w_vis.flatten()
        
#         # Compute statistics
#         min_val = w_flat.min()
#         max_val = w_flat.max() 
#         mean_val = w_flat.mean()
#         median_val = np.median(w_flat)
        
#         # Set threshold to median if not provided (better for small values)
#         if threshold is None:
#             threshold = median_val
        
#         print(f"Attention stats - Min: {min_val:.2e}, Max: {max_val:.2e}, Mean: {mean_val:.2e}, Median: {median_val:.2e}")
        
#         # Create separate arrays for values above and below threshold
#         high_values = w_flat[w_flat > threshold]
#         low_values = w_flat[w_flat <= threshold]
#         low_values = low_values
        
#         # Create the histograms plot
#         plt.figure(figsize=(18, 8))
        
#         # First subplot - histogram of values > threshold - use log scale if needed
#         plt.subplot(1, 2, 1)
#         if len(high_values) > 0:
#             plt.hist(high_values, bins=30, color='red', alpha=0.7)
#             plt.axvline(x=threshold, color='black', linestyle='--', 
#                     label=f'Threshold: {threshold:.2e}')
#             plt.title(f"Attention values > {threshold:.2e} (count: {len(high_values)})")
#             plt.xlabel("Attention value")
#             plt.ylabel("Frequency")
            
#             # Use log scale for y-axis if there are large variations in counts
#             if np.max(np.histogram(high_values, bins=30)[0]) / np.min(np.histogram(high_values, bins=30)[0] + 1) > 100:
#                 plt.yscale('log')
                
#             plt.legend()
#         else:
#             plt.text(0.5, 0.5, f"No values > {threshold:.2e}", 
#                     ha='center', va='center', fontsize=14)
        
#         # Second subplot - histogram of values <= threshold
#         plt.subplot(1, 2, 2)
#         if len(low_values) > 0:
#             plt.hist(low_values, bins=30, color='blue', alpha=0.7)
#             # plt.axvline(x=threshold, color='black', linestyle='--',
#             #         label=f'Threshold: {threshold:.2e}')
#             plt.title(f"Attention values <= {threshold:.2e} (count: {len(low_values)})")
#             plt.xlabel("Attention value")
#             plt.ylabel("Frequency")
            
#             # Use log scale for y-axis if there are large variations in counts
#             if np.max(np.histogram(low_values, bins=30)[0]) / np.min(np.histogram(low_values, bins=30)[0] + 1) > 100:
#                 plt.yscale('log')
                
#             plt.legend()
#         else:
#             plt.text(0.5, 0.5, f"No values <= {threshold:.2e}", 
#                     ha='center', va='center', fontsize=14)
        
#         # Additional statistics with scientific notation for small numbers
#         plt.figtext(0.5, 0.01, 
#                 f"Statistics: Mean={mean_val:.2e}, Median={median_val:.2e}, "
#                 f"Min={min_val:.2e}, Max={max_val:.2e}", 
#                 ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
#         # Save the plot
#         save_path = f"/home/csgrad/susimmuk/CSE676/NFormer/visualizations/{name}_sample{sample_idx}_head{head_idx}_at_layer_{l}_original.png"
#         plt.tight_layout()
#         plt.savefig(save_path)
#         plt.close()
# #         # print(f"Saved histogram visualization to {save_path}")

#     def rns(self, l, w, v, rns_indices):
#         bs,hn,dl,_ = w.shape
#         rns_indices = rns_indices.unsqueeze(1).repeat(1,hn,1,1)
#         mask = torch.zeros_like(w).scatter_(3, rns_indices,torch.ones_like(rns_indices, dtype=w.dtype))
#         mask = mask * mask.transpose(2,3)
#         if 'cuda' in str(w.device):
#             mask = mask.cuda()
#         else:
#             mask = mask.cpu()
#         w = w * mask + -1e9 * (1 - mask)
#         w = F.softmax(w,dim=3)
#         # print(w.max(), w.min())
#         self.visualize_threshold_attention(w, "rns_headwise", l, threshold=0.5, sample_idx=0, head_idx=0)
#         self.visualize_threshold_attention(w, "rns_headwise", l, threshold=0.5, sample_idx=0, head_idx=1)
#         a_v = torch.matmul(w, v)
#         return a_v


#     def merge_heads(self, x):
#         x = x.permute(0, 2, 1, 3).contiguous()
#         new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
#         return x.view(*new_x_shape)

#     def split_heads(self, x, k=False):
#         new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
#         x = x.view(*new_x_shape)
#         if k:
#             return x.permute(0, 2, 3, 1)
#         else:
#             return x.permute(0, 2, 1, 3)


#     def forward(self, l,  x, num_landmark, rns_indices):
#         x = self.c_attn(x)
#         query, key, value = x.split(self.split_size, dim=2)
#         query = self.split_heads(query)
#         key = self.split_heads(key, k=True)
#         value = self.split_heads(value)
#         mask = None
#         a = self._attn(l, query, key, value, num_landmark, rns_indices)
#         a = self.merge_heads(a)
#         a = self.c_proj(a)
#         a = self.resid_dropout(a)
#         return a


# class MLP(nn.Module):
#     def __init__(self, n_state, cfg):
#         super(MLP, self).__init__()
#         nx = cfg.MODEL.N_EMBD
#         self.c_fc = Conv1D(n_state, 1, nx)
#         self.c_proj = Conv1D(nx, 1, n_state)
#         self.act = ACT_FNS[cfg.MODEL.AFN]
#         self.dropout = nn.Dropout(cfg.MODEL.RESID_PDROP)

#     def forward(self, x):
#         h = self.act(self.c_fc(x))
#         h2 = self.c_proj(h)
#         return self.dropout(h2)


# class Block(nn.Module):
#     def __init__(self, n_ctx, cfg, scale=False):
#         super(Block, self).__init__()
#         nx = cfg.MODEL.N_EMBD
#         self.attn = Attention(nx, n_ctx, cfg, scale)
#         self.ln_1 = LayerNorm(nx)
#         self.mlp = MLP(4 * nx, cfg)
#         self.ln_2 = LayerNorm(nx)

#     def forward(self, l, x, num_landmark, rns_indices):
#         a = self.attn(l, x, num_landmark, rns_indices)
#         n = self.ln_1(x + a)
#         m = self.mlp(n)
#         h = self.ln_2(n + m)
#         return h


# class NFormer(nn.Module):
#     """ NFormer model """

#     def __init__(self, cfg, vocab=40990, n_ctx=1024, num_classes = 751):
#         super(NFormer, self).__init__()
#         self.num_classes = num_classes
        
#         block = Block(n_ctx, cfg, scale=True)
#         self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.MODEL.N_LAYER)])

#         self.bottleneck = nn.BatchNorm1d(cfg.MODEL.N_EMBD)
#         self.bottleneck.bias.requires_grad_(False)  # no shift
#         self.bottleneck.apply(weights_init_kaiming)
        
#         self.classifier = nn.Linear(cfg.MODEL.N_EMBD, self.num_classes, bias=False)
#         self.classifier.apply(weights_init_classifier)
#         self.topk = cfg.MODEL.TOPK
#         self.num_landmark = cfg.MODEL.LANDMARK

#     def forward(self, x):
#         _, rns_indices = torch.topk(torch.bmm(x/torch.norm(x,p=2,dim=2,keepdim=True),(x/torch.norm(x,p=2,dim=2,keepdim=True)).transpose(1,2)), self.topk, dim=2)
#         for l, block in enumerate(self.h):
#             x = block(l, x, self.num_landmark, rns_indices)

#         bs,dl,d = x.shape
#         x = x.reshape(bs*dl,d)
#         feat = self.bottleneck(x)
#         cls_score = self.classifier(feat)
#         x = x.reshape(bs,dl,d)
#         feat = feat.reshape(bs,dl,d)
#         cls_score = cls_score.reshape(bs,dl,-1)

#         if self.training:
#             return cls_score, x
#         else:
#             return feat
