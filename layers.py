import torch
import torch.nn as nn

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
                        nmagic[i] = sum([2*k-N-1/(k*1.0) for k in list(range(i+1, N+1))])
                x_ = x[l][idv, :, :, :]
                prod = x_*(np.reshape(nmagic, (N, 1, 1, 1)))
                sprod = np.sum(prod, axis=0)
                out_dynamic_images[v, :, :, :] = sprod
            result.append(out_dynamic_images)
        result = np.array(result, dtype='float32')
        return torch.Tensor(result)

    def backward(self, grad_output):
        x, vidids, = self.saved_tensors
        backpropogated_dynamic_images = np.zeros(x.size())
        nvids = np.max(vidids.numpy())
        vidids = vidids.numpy().tolist()
        result = []
        for g in grad_output:
            for v in range(nvids):
                idv = [i for i in range(len(vidids)) if vidids[i] == v]
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
        return torch.Tensor(result)
