import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

class ApproximateRankPooling(torch.autograd.Function):

    def __init__(self):
        super(ApproximateRankPooling, self).__init__()

    def forward(self, x, vidids):
        nvids = np.max(vidids.numpy())
        out_size_fwd = np.insert(np.array(x.size()[2:]), 0, nvids, axis=0).tolist()

        result = []
        out_dynamic_images = np.zeros(out_size_fwd)
        self.save_for_backward(x, vidids)
        x = x.numpy()
        vidids = vidids.numpy()
        for l in range(len(x)):
            for v in range(nvids):
                idv = [i for i in range(len(vidids[l])) if vidids[l][i] == v]
                N = len(idv)
                nmagic = np.zeros(N)
                if N == 1:
                    nmagic = 1
                else:
                    for i in range(N):
                        nmagic[i] = np.sum([(2*k-N-1)/(k*1.0) for k in list(range(i+1, N+1))])
                x_ = x[l][idv, :, :, :]
                prod = x_*(np.reshape(nmagic, (N, 1, 1, 1)))
                sprod = np.sum(prod, axis=0)
                out_dynamic_images[v, :, :, :] = sprod
            result.append(out_dynamic_images)
        result = np.array(result, dtype='float32')
        return torch.from_numpy(result)

    def backward(self, grad_output):
        x, vidids, = self.saved_tensors
        backpropogated_dynamic_images = np.zeros(x.size())
        nvids = np.max(vidids.numpy())
        vids = vidids.numpy().tolist()
        result = []
        for g in grad_output:
            for v in range(nvids):
                idv = [i for i in range(len(vids)) if vids[i] == v]
                N = len(idv)
                nmagic = np.zeros(N)
                if N == 1:
                    nmagic = 1
                else:
                    for i in range(N):
                        nmagic[i] = sum([2*k-N-1/(k*1.0) for k in list(range(i, N+1))])
                grad_input = g.clone().numpy()
                dzdy = np.tile(grad_input[v, :, :, :], [N, 1, 1, 1])
                backpropogated_dynamic_images[idv, :, :, :] = dzdy * (np.reshape(nmagic, (N, 1, 1, 1)))
            result.append(backpropogated_dynamic_images)
        result = np.array(result, dtype='float32')
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
        result = []
        for i in range(len(x)):
            shapex = x[i].shape
            if not np.all(np.array(shapex[2:]).flatten() == 1):
                x_ = np.reshape(x[i], (np.prod(shapex[2:]), -1))
            else:
                x_ = x[i]
            x_ = x_ + offset
            y = np.array(x_ * ((scale / np.sqrt(np.sum(x_ * x_))) + np.float32(1e-12)), dtype='float32')
            if np.all(np.logical_or(y[:] < clip[0], y[:] > clip[1])):
                print 'Too small clipping interval'
            y[y[:] < clip[0]] = clip[0]
            y[y[:] > clip[1]] = clip[1]
            if not np.all(np.array(shapex[2:]).flatten() == 1):
                y = np.reshape(y, shapex)
            result.append(y)
        result = np.array(result, dtype='float32')
        return torch.from_numpy(result)

    def backward(self, grad_output):
        x, params, = self.saved_tensors()
        grad_input = grad_output.clone()

        p = params.numpy().tolist()
        scale = p[0]
        clip = p[1:3]
        offset = p[3]

        result = []
        x = x.numpy()
        for i in range(len(grad_input)):
            shapex = x[i].shape
            x_ = []
            grad_input_ = []
            if not np.all(np.array(shapex[2:]).flatten() == 1):
                grad_input_ = np.reshape(grad_input[i], (np.prod(shapex[2:]), -1))
                x_ = np.reshape(x[i], (np.prod(shapex[2:]), -1))
            else:
                grad_input_ = grad_input[i]
                x_ = x[i]
            x_ = x_ + offset

            len_ = 1 / np.sqrt(np.sum(x_ * x_) + np.float32(1e-12))
            grad_input_i = grad_input_ * (np.power(len_, 3))
            y = scale * ((grad_input_ * len_) - (x_ * np.sum(x_ * grad_input_i)))
            if not np.all(np.array(shapex[2:]).flatten() == 1):
                y = np.reshape(y, shapex)
            result.append(y)
        result = np.array(result, dtype='float32')
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

    # TODO: Complete backward pass
    def backward(self, grad_output):
        x, = self.saved_tensors()

        grad_input = grad_output.clone()
        result = np.zeros(list(x.size()))

        pool_layer = nn.MaxPool2d((1, 9), stride=(1, 9))
        for i in range(len(grad_input)):
            inp = x[i].permute(1,2,3,0)
            grad_inp = grad_input[i].permute(1,2,3,0)

        return torch.from_numpy(result)
