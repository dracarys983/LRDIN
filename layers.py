import torch
import torch.nn as nn

import numpy as np

class ApproximateRankPooling(nn.Module):

    def __init__(self, x, nvids, dframes):
        super(ApproximateRankPooling, self).__init__()
        self.batch_images = x
        self.nvids = nvids
        self.dframes = dframes
        self.out_size_fwd = np.insert(np.array(x.size()[1:]), 0, nvids, axis=0).tolist()

    def forward(self, x):
        out_dynamic_images = np.zeros(self.out_size_fwd)
        x = x.numpy()
        for v in range(self.nvids):
            idv = [i for i in range(v*self.dframes,(v+1)*self.dframes)]
            nmagic = np.zeros(self.dframes)
            if self.dframes == 1:
                nmagic = 1
            else:
                for i in range(self.dframes):
                    nmagic[i] = sum([2*k-self.dframes-1/(k*1.0) for k in list(range(i,self.dframes+1))])
            x = x[idv, :, :, :]
            out_dynamic_images[v] = np.sum([x*(np.reshape(nmagic,(self.dframes,1,1,1)))], axis=0)
        return torch.Tensor(out_dynamic_images)

    def backward(self):
        propogated_dynamic_images = np.zeros(self.batch_images.size())
