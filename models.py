import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

import numpy as np

import layers

class DINet(nn.Module):

    def __init__(self, orig_model, arch, num_classes):
        super(DINet, self).__init__()

        if arch.startswith('alexnet'):
            self.arpool = layers.ARPool()
            self.l2norm = layers.L2Norm()
            self.features = orig_model.features
            self.temppool = layers.TempPool()
            self.classifier = nn.Sequential(nn.Dropout(),
                                            nn.Linear(256 * 6 * 6, 4096),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(4096, num_classes),
                                            nn.LogSoftmax()
                                            )
            self.modelName = 'alexnet'
        else:
            raise NotImplementedError()

    def forward(self, x, vidids):
        params = [6e3, -128, 128, 0]
        params = Variable(torch.from_numpy(np.array(params)).cuda())
        # Initialize the result tensor
        result = Variable(torch.cuda.FloatTensor(x.size(0), 101))
        b = 0
        # Forward pass through Alexnet
        for batch in x:
            # Approxmiate Rank Pooling layer : Get Dynamic Images
            dyn = self.arpool(batch, vidids[b])
            # L2Normalize layer : Normalize the Dynamic Images
            nimgs = self.l2norm(dyn, params)
            # Debug code: Save and check dynamic images formed
            # fname = 'dynamic_image_' + str(b) + '.jpg'
            # save_image(tensor=nimgs.data, filename=fname)
            # Convolutional layers (get the features)
            f = self.features(nimgs)
            # TemporalPooling layer : Pool across Dynamic Images
            t = self.temppool(f)
            t = t.contiguous().view(t.size(0), 256 * 6 * 6)
            # Classification layers (Fully Connected) with LogSoftmax
            c = self.classifier(t)
            result[b, :] = c[0]
            b += 1
        return result
