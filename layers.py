import torch
import torch.nn as nn

import numpy as np

class ApproximateRankPooling(nn.Module):

    def __init__(self, x, nvids, dframes, vidids):
        super(ApproximateRankPooling, self).__init__()
        self.batch_images = x
        self.nvids = nvids
        self.dframes = dframes
        self.vidids = vidids
        self.out_size_fwd = np.insert(np.array(x.size()[1:]), 0, nvids, axis=0).tolist()

    def forward(self, x):
        out_dynamic_images = np.zeros(self.out_size_fwd)
        x = x.numpy()
        for v in range(self.nvids):
            idv = [i for i in range(len(self.vidids)) if self.vidids[i] == v]
            N = len(idv)
            nmagic = np.zeros(N)
            if N == 1:
                nmagic = 1
            else:
                for i in range(N):
                    nmagic[i] = sum([2*k-N-1/(k*1.0) for k in list(range(i,N+1))])
            x = x[idv, :, :, :]
            out_dynamic_images[v, :, :, :] = np.sum([x*(np.reshape(nmagic,(N,1,1,1)))], axis=0)
        return torch.Tensor(out_dynamic_images)

    def backward(self):
        backpropogated_dynamic_images = np.zeros(self.batch_images.size())
        for v in range(self.nvids):
            idv = [i for i in range(len(self.vidids)) if self.vidids[i] == v]
            N = len(idv)
            nmagic = np.zeros(N)
            if N == 1:
                nmagic = 1
            else:
                for i in range(N):
                    nmagic[i] = sum([2*k-N-1/(k*1.0) for k in list(range(i,N+1))])
            dzdy = np.tile(out_dynamic_images[v, :, :, :], [N, 1, 1, 1])
            backpropogated_dynamic_images[idv, :, :, :] = dzdy * (np.reshape(nmagic, (N,1,1,1)))
        return torch.Tensor(backpropogated_dynamic_images)
