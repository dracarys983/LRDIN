import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

class ApproximateRankPooling(torch.autograd.Function):

    def __init__(self):
        super(ApproximateRankPooling, self).__init__()

    def forward(self, x, vidids):
        nvids = torch.cuda.LongTensor([[torch.max(vidids)]])
        sz = torch.cuda.LongTensor([x.size()[1:]])
        out_size_fwd = torch.cat((nvids, sz), 1)

        out_dynamic_images = torch.cuda.FloatTensor(torch.Size(out_size_fwd.tolist()[0]))
        self.save_for_backward(x, vidids)
        for v in range(nvids[0][0]):
            idv = [i for i in range(vidids.size()[0]) if vidids[i] == v]
            N = len(idv)
            nmagic = np.zeros(N, dtype='float32')
            if N == 1:
                nmagic = 1
            else:
                for i in range(N):
                    nmagic[i] = np.sum([(2*(k+1)-N-1)/((k+1)*1.0) for k in list(range(i, N+1))])
            indices = torch.cuda.LongTensor(idv)
            x_ = torch.index_select(x, 0, indices)
            nmagic_ex = torch.cuda.FloatTensor(x_.size())
            for i in range(N):
                nmagic_ex[i].fill_(float(nmagic[i]))
            prod = torch.mul(x_, nmagic_ex)
            sprod = torch.sum(prod, dim=0)
            out_dynamic_images[v, :, :, :] = sprod
        return out_dynamic_images

    def backward(self, grad_input):
        x, vidids, = self.saved_tensors
        backpropogated_dynamic_images = torch.cuda.FloatTensor(x.size())
        nvids = torch.max(vidids)
        for v in range(nvids):
            idv = [i for i in range(vidids.size()[0]) if vidids[i] == v]
            N = len(idv)
            nmagic = np.zeros(N, dtype='float32')
            if N == 1:
                nmagic = 1
            else:
                for i in range(N):
                    nmagic[i] = sum([2*(k+1)-N-1/((k+1)*1.0) for k in list(range(i, N+1))])
            sz = [N]
            sz.extend(list(x.size()[1:]))
            nmagic_ex = torch.cuda.FloatTensor(torch.Size(sz))
            for i in range(N):
                nmagic_ex[i].fill_(float(nmagic[i]))
            dzdy = grad_input[v, :, :, :]
            dzdy = dzdy.repeat(N, 1, 1, 1)
            for i in range(len(idv)):
                backpropogated_dynamic_images[idv[i],:,:,:] = torch.mul(dzdy[i,:,:,:], nmagic_ex[i,:,:,:])
        return backpropogated_dynamic_images, vidids


class L2Normalize(torch.autograd.Function):

    def __init__(self):
        super(L2Normalize, self).__init__()

    def forward(self, x, params):
        self.save_for_backward(x, params)
        scale = params[0]
        clip = params[1:3]
        offset = params[3]

        shapex = x.size()
        check = torch.cuda.ByteTensor(len(shapex[1:])).fill_(1)
        if not torch.equal(torch.eq(torch.cuda.IntTensor([shapex[1:]]), 1), check):
            x_ = x.contiguous().view(torch.Size([torch.prod(torch.cuda.IntTensor([shapex[1:]])), -1]))
        else:
            x_ = x
        x_ = x_ + offset

        x_sum = torch.cuda.FloatTensor([torch.sum(torch.mul(x_, x_))])
        x_prod = torch.mul(x_, scale / torch.sqrt(x_sum)[0])
        check = torch.cuda.ByteTensor(x_prod.size()).fill_(1)
        less = torch.lt(x_prod, clip[0])
        great = torch.gt(x_prod, clip[1])
        if torch.equal((less | great), check):
            print 'Too small clipping interval'

        x_prod = torch.clamp(x_prod, min=clip[0], max=clip[1])
        if not torch.equal(torch.eq(torch.cuda.IntTensor([shapex[1:]]), 1), check):
            x_prod = x_prod.contiguous().view(shapex)
        return x_prod

    def backward(self, grad_input):
        x, params, = self.saved_tensors

        scale = params[0]
        clip = params[1:3]
        offset = params[3]

        shapex = x.size()
        x_ = []
        grad_input_ = []
        check = torch.cuda.ByteTensor(len(shapex[1:])).fill_(1)
        if not torch.equal(torch.eq(torch.cuda.IntTensor([shapex[1:]]), 1), check):
            grad_input_ = grad_input.contiguous().view(torch.Size([torch.prod(torch.cuda.IntTensor([shapex[1:]])), -1]))
            x_ = x.contiguous().view(torch.Size([torch.prod(torch.cuda.IntTensor([shapex[1:]])), -1]))
        else:
            grad_input_ = grad_input
            x_ = x
        x_ = x_ + offset

        x_sum = torch.cuda.FloatTensor([torch.sum(torch.mul(x_, x_))])
        len_ = 1 / torch.sqrt(x_sum)[0]
        grad_input_i = grad_input_ * (np.power(len_, 3))
        y = scale * ((grad_input_ * len_) - (x_ * torch.sum(x_ * grad_input_i)))
        if not torch.equal(torch.eq(torch.cuda.IntTensor([shapex[1:]]), 1), check):
            y = y.contiguous().view(shapex)
        return y, params


class TemporalPooling(torch.autograd.Function):

    def __init__(self):
        super(TemporalPooling, self).__init__()

    def forward(self, x):
        self.save_for_backward(x)

        last_dim = x.size()[0]
        pool_layer = nn.MaxPool2d((1, last_dim), stride=(1, last_dim))
        inp = Variable(x.permute(1,2,3,0))
        y = pool_layer(inp)
        return y.data.permute(3,0,1,2)

    def backward(self, grad_input):
        x, = self.saved_tensors

        last_dim = x.size()[0]

        inp = x.permute(1,2,3,0)
        grad_inp = grad_input.permute(1,2,3,0)
        result = torch.cuda.FloatTensor(inp.size())

        N, C, H, W = inp.size()
        H_ = 1 + (H - 1) / 1
        W_ = 1 + (W - last_dim) / last_dim
        for n in range(N):
            for c in range(C):
                for h in range(H_):
                    for w in range(W_):
                        h1 = h
                        h2 = h + 1
                        w1 = w * last_dim
                        w2 = w * last_dim + last_dim
                        window = inp[n, c, h1:h2, w1:w2]
                        window2 = window.contiguous().view(1*last_dim)
                        window3 = torch.cuda.FloatTensor(window2.size())
                        value, index = torch.max(window2, 0)
                        window3[index] = 1

                        result[n, c, h1:h2, w1:w2] = window3.contiguous().view(torch.Size([1, last_dim])) * grad_inp[n,c,h,w]
        return result.permute(3,0,1,2)


class ARPool(nn.Module):
    def __init__(self):
        super(ARPool, self).__init__()
        self.layer = ApproximateRankPooling()

    def forward(self, x, vidids):
        result = self.layer(x, vidids)
        return result

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.layer = L2Normalize()

    def forward(self, x, params):
        result = self.layer(x, params)
        return result

class TempPool(nn.Module):
    def __init__(self):
        super(TempPool, self).__init__()
        self.layer = TemporalPooling()

    def forward(self, x):
        result = self.layer(x)
        return result
