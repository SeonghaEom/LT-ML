import torch
import numpy as np
import torch.nn as nn
from voc import *
from coco import *
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
from config import seed_everything
seed_everything(0)
import timm

# from models import *
from backbones.config import config
import pathlib
from torch import nn, einsum
from einops import rearrange, repeat


class inter_attention(nn.Module):
  def __init__(self, q_dim, kv_dim, inner_dim):
    super(inter_attention, self).__init__()

    group_queries = True
    group_key_values = True
    offset_groups = 1
    self.heads=1
    self.to_q = nn.Conv1d(q_dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
    self.to_k = nn.Conv1d(kv_dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
    self.to_v = nn.Conv1d(kv_dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
    self.to_out = nn.Conv1d(inner_dim, q_dim, 1)


  def get_attention( self, query, kv):
    k, v = self.to_k(kv), self.to_v(kv)
    # k.shape, v.shape #(torch.Size([1, 196, 768]), torch.Size([1, 196, 768]))
    q = self.to_q(query)
    # q.shape #torch.Size([1, 196, 768])

    # split out heads
    q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = self.heads), (q, k, v))

    # query / key similarity
    sim = einsum('b h i d, b h j d -> b h i j', q, k)
    # numerical stability

    sim = sim - sim.amax(dim = -1, keepdim = True).detach()

    # attention
    dropout = nn.Dropout(0.0)
    attn = sim.softmax(dim = -1)
    attn = dropout(attn)

    # aggregate and combine heads
    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    out = rearrange(out, 'b h n d -> b (h d) n')
    out = self.to_out(out)
    # print(out.shape)
    return out