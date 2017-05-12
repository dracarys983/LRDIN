import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

class ApproximateRankPooling(torch.autograd.Function):

    def __init__(self):
        super(ApproximateRankPooling, self).__init__()

    def forward(self, x, vidids):
        nvids = np.max(vidids.numpy())
        out_size_fwd = np.insert(np.array(x.size()[1:]), 0, nvids, axis=0).tolist()

        out_dynamic_images = np.zeros(out_size_fwd)
        self.save_for_backward(x, vidids)
        x = x.numpy()
        vidids = vidids.numpy()
        for v in range(nvids):
            idv = [i for i in range(len(vidids)) if vidids[i] == v]
            N = len(idv)
            nmagic = np.zeros(N)
            if N == 1:
                nmagic = 1
            else:
                for i in range(N):
                    nmagic[i] = np.sum([(2*k-N-1)/(k*1.0) for k in list(range(i+1, N+1))])
            x_ = x[idv, :, :, :]
            prod = x_*(np.reshape(nmagic, (N, 1, 1, 1)))
            sprod = np.sum(prod, axis=0)
            out_dynamic_images[v, :, :, :] = sprod
        result = np.array(out_dynamic_images, dtype='float32')
        return torch.from_numpy(result)

    def backward(self, grad_output):
        print 'ApproximateRankPooling backward pass'
        x, vidids, = self.saved_tensors
        backpropogated_dynamic_images = np.zeros(x.size())
        nvids = np.max(vidids.numpy())
        vids = vidids.numpy().tolist()
        for v in range(nvids):
            idv = [i for i in range(len(vids)) if vids[i] == v]
            N = len(idv)
            nmagic = np.zeros(N)
            if N == 1:
                nmagic = 1
            else:
                for i in range(N):
                    nmagic[i] = sum([2*k-N-1/(k*1.0) for k in list(range(i, N+1))])
            grad_input = grad_output.clone().numpy()
            dzdy = np.tile(grad_input[v, :, :, :], [N, 1, 1, 1])
            backpropogated_dynamic_images[idv, :, :, :] = dzdy * (np.reshape(nmagic, (N, 1, 1, 1)))
        result = np.array(backpropogated_dynamic_images, dtype='float32')
        return torch.from_numpy(result), vidids


class L2Normalize(torch.autograd.Function):

    def __init__(self):
        super(L2Normalize, self).__init__()

    def forward(self, x, params):
        self.save_for_backward(x, params)
        params = params.numpy().tolist()
        scale = params[0]
        clip = params[1:3]
        offset = params[3]

        x = x.numpy()
        shapex = x.shape
        if not np.all(np.array(shapex[1:]).flatten() == 1):
            x_ = np.reshape(x, (np.prod(shapex[1:]), -1))
        else:
            x_ = x
        x_ = x_ + offset
        y = np.array(x_ * ((scale / np.sqrt(np.sum(x_ * x_))) + np.float32(1e-12)), dtype='float32')
        if np.all(np.logical_or(y[:] < clip[0], y[:] > clip[1])):
            print 'Too small clipping interval'
        y[y[:] < clip[0]] = clip[0]
        y[y[:] > clip[1]] = clip[1]
        if not np.all(np.array(shapex[1:]).flatten() == 1):
            y = np.reshape(y, shapex)
        result = np.array(y, dtype='float32')
        return torch.from_numpy(result)

    def backward(self, grad_output):
        print 'L2Normalize backward pass'
        x, params, = self.saved_tensors
        grad_input = grad_output.clone().numpy()

        p = params.numpy().tolist()
        scale = p[0]
        clip = p[1:3]
        offset = p[3]

        x = x.numpy()
        shapex = x.shape
        x_ = []
        grad_input_ = []
        if not np.all(np.array(shapex[1:]).flatten() == 1):
            grad_input_ = np.reshape(grad_input, (np.prod(shapex[1:]), -1))
            x_ = np.reshape(x, (np.prod(shapex[1:]), -1))
        else:
            grad_input_ = grad_input
            x_ = x
        x_ = x_ + offset

        len_ = 1 / np.sqrt(np.sum(x_ * x_) + np.float32(1e-12))
        grad_input_i = grad_input_ * (np.power(len_, 3))
        y = scale * ((grad_input_ * len_) - (x_ * np.sum(x_ * grad_input_i)))
        if not np.all(np.array(shapex[1:]).flatten() == 1):
            y = np.reshape(y, shapex)
        result = np.array(y, dtype='float32')
        return torch.from_numpy(result), params


class TemporalPooling(torch.autograd.Function):

    def __init__(self):
        super(TemporalPooling, self).__init__()

    def forward(self, x):
        self.save_for_backward(x)

        pool_layer = nn.MaxPool2d((1, 9), stride=(1, 9))
        inp = Variable(x.permute(1,2,3,0))
        y = pool_layer(inp)
        return y.data.permute(3,0,1,2)

    def backward(self, grad_output):
        print 'TemporalPooling backward pass'
        x, = self.saved_tensors

        grad_input = grad_output.clone()

        inp = x.permute(1,2,3,0).numpy()
        grad_inp = grad_input.permute(1,2,3,0).numpy()
        result = np.zeros(inp.shape, dtype='float32')

        N, C, H, W = inp.shape
        H_ = 1 + (H - 1) / 1
        W_ = 1 + (W - 9) / 9
        for n in range(N):
            for c in range(C):
                for h in range(H_):
                    for w in range(W_):
                        h1 = h
                        h2 = h + 1
                        w1 = w * 9
                        w2 = w * 9 + 9
                        window = inp[n, c, h1:h2, w1:w2]
                        window2 = np.reshape(window, 1*9)
                        window3 = np.zeros_like(window2)
                        window3[np.argmax(window2)] = 1

                        result[n, c, h1:h2, w1:w2] = np.reshape(window3, (1, 9)) * grad_inp[n,c,h,w]

        print 'TemporalPooling backward pass done'
        return torch.from_numpy(result).permute(3,0,1,2)


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
