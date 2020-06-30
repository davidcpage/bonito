"""
Bonito Model template
"""
from collections import OrderedDict as odict
from functools import partial

import torch
import torch.nn as nn
from torch import sigmoid

from torch.nn import ReLU, LeakyReLU
from torch.nn import Module, ModuleList, Sequential, Conv1d, BatchNorm1d, Dropout

from fast_ctc_decode import beam_search, viterbi_search

@torch.jit.script
def _swish_fwd_bwd(x): 
    sig = torch.sigmoid(x)
    return x * sig, sig * (1. + x * (1. - sig))

class _Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        fwd, bwd = _swish_fwd_bwd(x)
        ctx.save_for_backward(bwd)
        return fwd

    @staticmethod
    def backward(ctx, grad):
        bwd, = ctx.saved_tensors
        return bwd * grad

swish = _Swish.apply

class Swish(nn.Module):
    """
    Swish Activation function

    https://arxiv.org/abs/1710.05941
    """
    def forward(self, x):
        return swish(x)

activations = {
    "relu": ReLU,
    "swish": Swish,
}


def _gauss(x, mu, sig):
    return (1./sig)*torch.exp(-(((x-mu[:, None])**2)/(2*sig**2)))

@torch.jit.script
def _diff_gauss(x, mu, sig1, sig2):
    return 0.5*(_gauss(x, mu, sig1) - _gauss(x, mu, sig2))

class Embed(torch.nn.Module):
    def __init__(self, sig1=0.2, sig2=None, size=64, range_min=-2.2, range_max=2.2):
        super().__init__()
        self.size=size
        self.sig1, self.sig2 = [nn.Parameter(torch.tensor(x), requires_grad=False) for x in (sig1, (sig2 or 2*sig1))]
        self.ys = nn.Parameter(torch.linspace(range_min, range_max, size), requires_grad=False)
    def forward(self, x):
        return _diff_gauss(x, self.ys, self.sig1, self.sig2)

class Add(nn.Module):
    def forward(self, x, y): return x + y

class SplitMerge(nn.Module):
    def __init__(self, branches, merge=Add(), pre=nn.Identity(), post=None):
        super().__init__()
        if isinstance(branches, list):
            branches = odict([('branch_%s'%i, branch) for i, branch in enumerate(branches)])
        for name, branch in branches.items():
            self.add_module(name, branch)
        self.branches = branches
        self.merge, self.pre, self.post = merge, pre, post 

    def forward(self, x):
        x = self.pre(x)
        branch_outputs = [branch(x) for branch in self.branches.values()]
        if self.merge is not None:
            x = self.merge(*branch_outputs)
        if self.post is not None:
            x = self.post(x)
        return x

    def to_graph(self):
        graph = odict([('pre', self.pre)] + [(k, (v, ['pre'])) for k,v in self.branches.items()])
        if self.merge:
            graph[type(self.merge).__name__] = (self.merge, list(self.branches.keys()))
        if self.post:
            graph['post'] = self.post
        return graph

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dims = dim0, dim1
    def forward(self, x):
        return x.transpose(*self.dims)

class PixelShuffle1d(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        
    def forward(self, x):
        N, C, T = x.shape
        x = x.view(N, C//self.upscale_factor, self.upscale_factor, T)
        return x.transpose(2, 3).reshape(N, C//self.upscale_factor, T*self.upscale_factor)
    
def sequential(layers, names=None):
    if names is not None:
        return sequential(odict(zip(names, layers.values() if isinstance(layers, odict) else layers)))
    if isinstance(layers, (list, tuple)):
        layers = (sequential(v) for v in layers if v is not None)
        return nn.Sequential(odict(('{}_{}'.format(type(v).__name__, i), v) for i, v in enumerate(layers)))
    elif isinstance(layers, odict):
        return nn.Sequential(odict(((k, sequential(v)) for (k, v) in layers.items() if v is not None)))
    return layers

class ResBlock(SplitMerge): pass

def residual(layers, shortcut=nn.Identity(), merge=Add(), pre=nn.Identity(), post=None):
    return ResBlock(
        branches=odict([
            ('main', sequential(layers)), 
            ('shortcut', sequential(shortcut))
        ]), 
        merge=sequential(merge), 
        pre=sequential(pre),
        post=sequential(post)
    )

def conv(c_in, c_out, ks, stride=1, bias=False, dilation=1, groups=1):
    if stride > 1 and dilation > 1:
        raise ValueError("Dilation and stride can not both be greater than 1")
    return nn.Conv1d(c_in, c_out, ks, stride=stride, padding=(ks // 2)*dilation, bias=bias, dilation=dilation, groups=groups)

class TCSConv1d(nn.Sequential): pass

def tcs_conv(c_in, c_out, ks, stride=1, bias=False, dilation=1):
    return TCSConv1d(odict([
        ('depthwise', conv(c_in, c_in, ks, stride=stride, bias=bias, dilation=dilation, groups=c_in)),
        ('pointwise', conv(c_in, c_out, 1, stride=1, bias=bias, dilation=dilation)),
    ]))

def bn(c, eps=1e-3, momentum=0.1):
    return nn.BatchNorm1d(c, eps=eps, momentum=momentum)

def dropout(p, inplace=True):
    return nn.Dropout(p, inplace=inplace) if p else None
            
def conv_layer(c_in, c_out, ks, stride=1, dilation=1, drop_p=0.0, act=Swish(), conv_type=conv):
    return [conv_type(c_in, c_out, ks, dilation=dilation, stride=stride), bn(c_out), act, dropout(drop_p)]

def conv_block(c_in, c_out, ks, stride=1, dilation=1, drop_p=0.0, act=Swish(), conv_type=conv, repeat=1):
    return [conv_layer(c_in, c_out, ks, stride=stride, dilation=dilation, drop_p=drop_p, act=act, conv_type=conv_type), *(
        conv_layer(c_out, c_out, ks, stride=1, dilation=dilation, drop_p=drop_p, act=act, conv_type=conv_type) for _ in range(repeat-1)
    )]

def res_block_A(c_in, c_out, ks, stride=1, dilation=1, drop_p=0.0, act=Swish(), conv_type=conv, repeat=1):
    block = conv_block(c_in, c_out, ks, dilation=dilation, stride=stride, drop_p=drop_p, act=act, conv_type=conv_type, repeat=repeat)
    block[-1], post = block[-1][:2], block[-1][2:]
    return residual(layers=block, shortcut=[conv(c_in, c_out, 1, dilation=dilation, stride=stride), bn(c_out)], post=post)

def res_block_B(c_in, c_out, ks, stride=1,  dilation=1, drop_p=0.0, act=Swish(),conv_type=conv, ks_pre=5, conv_type_pre=conv, repeat=1):
    layer = partial(conv_layer, drop_p=drop_p, act=act, conv_type=conv_type, dilation=dilation)
    return residual(
        pre = layer(c_in, c_out, ks_pre, stride, conv_type=conv_type_pre), 
        layers = [layer(c_out, c_out, ks, stride), *[layer(c_out, c_out, ks, 1) for _ in range(repeat-1)]],
    )

#FixMe: need a better way to handle down-up sampling
#when chunk size is not a multiple of stride
class _AddTrim(nn.Module):
    def forward(self, x, y):     
        return x[:, :, :y.size(2)] + y

def unet_block(c_in, c_out, ks, stride=1, dilation=1, drop_p=0.0, act=Swish(), conv_type=conv, ks_pre=5, conv_type_pre=conv, repeat=1):
    layer = partial(conv_layer, drop_p=drop_p, act=act, conv_type=conv_type, dilation=dilation)
    return residual(
        pre = layer(c_in, c_out, stride=stride, ks=ks_pre, conv_type=conv_type_pre),
        layers = [
            layer(c_out, 2*c_out, stride=2, ks=ks_pre, conv_type=conv_type_pre),
            [layer(2*c_out, 2*c_out, ks, 1) for _ in range(repeat)],
            PixelShuffle1d(2)
        ], 
        merge = _AddTrim()
    )

def ctc_decoder(layers):
    return sequential([layers, Transpose(1, 2), nn.LogSoftmax(2)], names=('layers', 'transpose', 'log_softmax'))

def block_type(block_config):
    block_types = (conv_block, res_block_A, res_block_B, unet_block)
    if 'block_type' in block_config: 
        return block_types[block_config['block_type']]
    return block_types[int(block_config['residual'])]

def conv_type(block_config):
    return tcs_conv if block_config['separable'] else conv

def encoder(config):
    if config['input'].get('embed', False):
        blocks, c_in = [Embed(sig1=0.2, size=64)], 64
    else:
        blocks, c_in = [], 1
    act = activations[config['encoder']['activation']]
    for b in config['block']:
        b = {k: (v[0] if isinstance(v, list) else v) for k,v in b.items()}
        blocks.append(
            block_type(b)(
                c_in=c_in, c_out=b['filters'], ks=b['kernel'], stride=b['stride'], 
                repeat=b['repeat'], drop_p=b['dropout'], dilation=b['dilation'], 
                act=act(), conv_type=conv_type(b)                                                               
            )
        )
        c_in = b['filters']
    return sequential(blocks)

class Model(nn.Sequential):
    """
    Model template for QuartzNet style architectures

    https://arxiv.org/pdf/1910.10261.pdf
    """
    def __init__(self, config):
        if 'qscore' not in config:
            self.qbias = 0.0
            self.qscale = 1.0
        else:
            self.qbias = config['qscore']['bias']
            self.qscale = config['qscore']['scale']

        self.config = config
        self.stride = config['block'][0]['stride'][0]
        self.alphabet = config['labels']['labels']
        self.features = config['block'][-1]['filters']
        self.config = config
        
        super().__init__(odict(
                [
                    ('encoder', encoder(config)), 
                    ('decoder', ctc_decoder(conv(self.features, len(self.alphabet), ks=1, bias=True)))
                ]
        ))

    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        if beamsize == 1 or qscores:
            seq, path  = viterbi_search(x, self.alphabet, qscores, self.qscale, self.qbias)
        else:
            seq, path = beam_search(x, self.alphabet, beamsize, threshold)
        if return_path: return seq, path
        return seq
