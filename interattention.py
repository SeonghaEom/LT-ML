import torch
import torch.nn as nn
from voc import *
from coco import *
from config import seed_everything
seed_everything(0)
from torch import nn, einsum
from einops import rearrange, repeat


class InterAttention(nn.Module):
  def __init__(self, q_dim, kv_dim, inner_dim, inter_dim, feature_dim=512):
    super(InterAttention, self).__init__()

    group_queries = True
    group_key_values = True
    offset_groups = 1
    self.heads=1
    self.to_q = nn.Conv1d(q_dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
    self.to_k = nn.Conv1d(kv_dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
    self.to_v = nn.Conv1d(kv_dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
    self.to_out = nn.Conv1d(inner_dim, q_dim, 1)
    self.ll = nn.ModuleList([
      nn.Linear(inter_dim[0], feature_dim, bias=False),
      nn.Linear(inter_dim[1], feature_dim, bias=False),
      nn.Linear(inter_dim[2], feature_dim, bias=False),
      nn.Linear(inter_dim[3], feature_dim, bias=False),
    ])
  def get_kv(self, int_li, tau=1):
    softmax = nn.functional.gumbel_softmax
    T = rearrange(int_li[0], 'b N D -> b D N')
    # print(T.shape)
    val= softmax(self.ll[0](int_li[0]), dim=1, tau=tau)
    # print(val.shape)
    res = torch.matmul(T, val)
    # res = rearrange(res, 'b D N -> b N D')
    # print(res.shape)

    T = rearrange(int_li[1], 'b N D -> b D N')
    # print(T.shape)
    val= softmax(self.ll[1](int_li[1]), dim=1, tau=tau)
    # print(val.shape)
    test = torch.matmul(T, val)
    # test = rearrange(test, 'b D N -> b N D')
    res = torch.concat((res, test), dim=1)
    # print(res.shape)

    T = rearrange(int_li[2], 'b N D -> b D N')
    # print(T.shape)
    val= softmax(self.ll[2](int_li[2]), dim=1, tau=tau)
    # print(val.shape)
    test = torch.matmul(T, val)
    # test = rearrange(test, 'b D N -> b N D')
    res = torch.concat((res, test), dim=1)
    # print(res.shape)

    T = rearrange(int_li[3], 'b N D -> b D N')
    # print(T.shape)
    val= softmax(self.ll[3](int_li[2]), dim=1, tau=tau)
    # print(val.shape)
    test = torch.matmul(T, val)
    # test = rearrange(test, 'b D N -> b N D')
    res = torch.concat((res, test), dim=1)
    # print(res.shape)

    return res

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
    dropout = nn.Dropout(0.2)
    attn = sim.softmax(dim = -1)
    attn = dropout(attn)

    # aggregate and combine heads
    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    out = rearrange(out, 'b h n d -> b (h d) n')
    out = self.to_out(out)
    # print(out.shape)
    return out