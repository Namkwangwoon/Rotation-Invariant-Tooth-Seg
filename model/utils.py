# Reference: https://github.com/qinzheng93/GeoTransformer

import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.utils import square_distance
import numpy as np

import re


def get_activation(activation, **kwargs):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        if 'negative_slope' in kwargs:
            negative_slope = kwargs['negative_slope']
        else:
            negative_slope = 0.01
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    elif activation == 'elu':
        return nn.ELU(inplace=True)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'gelu':
        return nn.GLU()
    else:
        raise RuntimeError('Activation function {} is not supported.'.format(activation))


class MonteCarloDropout(nn.Module):
    def __init__(self, p=0.1):
        super(MonteCarloDropout, self).__init__()
        self.p = p

    def forward(self, x):
        out = nn.functional.dropout(x, p=self.p, training=True)
        return out


def get_dropout(p, monte_carlo_dropout=False):
    if p is not None and p > 0:
        if monte_carlo_dropout:
            return MonteCarloDropout(p)
        else:
            return nn.Dropout(p)
    else:
        return None


def get_ftype(ftype):
    if ftype in ['out', 'fout', 'f_out', 'latent', 'logits', 'probs']:
        ptype = 'p_out'
        ftype = 'f_out' if ftype in ['out', 'fout'] else ftype
    elif ftype in ['sample', 'fsample', 'f_sample']:
        ptype = 'p_sample'
        ftype = 'f_sample' if ftype in ['sample', 'fsample'] else ftype
    else:
        raise KeyError(f'not supported ftype = {ftype}')
    return ftype, ptype


def parse_stage(stage, num_layers):
    stage = stage.replace('a', ''.join(f'{i}' for i in range(num_layers)))
    stage_list = [i.strip('_') for i in re.split('(\d+)', stage) if i and i.strip('_')]  # e.g. D012_U34
    assert len(stage_list) % 2 == 0, f'invalid stage compound: stage_list={stage_list} from stage={stage}'
    stage_n = [s for i, s in enumerate(stage_list) if i % 2 == 0]
    stage_i = [s for i, s in enumerate(stage_list) if i % 2 == 1]
    stage_list = [[(to_valid_stage(n), int(i)) for i in i_str] for n, i_str in zip(stage_n, stage_i)]
    stage_list = sum(stage_list, [])
    return stage_list


def fetch_pxo(stage_n, stage_i, stage_list, ftype):
    stage = stage_list[stage_n][stage_i]
    return stage['p_out'], stage[ftype], stage['offset']


def to_valid_stage(stage_n, short=False):
    if stage_n in ['D', 'down']:
        stage_n = 'D' if short else 'down'
    elif stage_n in ['U', 'up']:
        stage_n = 'U' if short else 'up'
    else:
        raise ValueError(f'invalid stage_n={stage_n}')
    return stage_n