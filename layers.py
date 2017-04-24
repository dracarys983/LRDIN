import torch
import torch.nn as nn

import numpy as np

class ApproximateRankPooling(nn.Module):

    def __init__(self):
        super(ApproximateRankPooling, self).__init__()
 
    def forward(self, x, vidids):
        nvids = max(vidids)
        out_size_fwd = np.insert(np.array(x.size()[1:]), 0, nvids, axis=0).tolist()

        out_dynamic_images = np.zeros(self.out_size_fwd)
        self.save_for_backward(x, torch.Tensor(vidids))
        x = x.numpy()
        for v in range(nvids):
            idv = [i for i in range(len(vidids)) if vidids[i] == v]
            N = len(idv)
            nmagic = np.zeros(N)
            if N == 1:
                nmagic = 1
            else:
                for i in range(N):
                    nmagic[i] = sum([2*k-N-1/(k*1.0) for k in list(range(i, N+1))])
            x = x[idv, :, :, :]
            out_dynamic_images[v, :, :, :] = np.sum([x*(np.reshape(nmagic, (N, 1, 1, 1)))], axis=0)
        return torch.Tensor(out_dynamic_images)

    def backward(self, grad_output):
        x, vidids, = self.saved_tensors
        backpropogated_dynamic_images = np.zeros(x.size())
        nvids = max(vidids)
        for v in range(nvids):
            idv = [i for i in range(len(vidids)) if vidids[i] == v]
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
        return torch.Tensor(backpropogated_dynamic_images)
