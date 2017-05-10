import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

import numpy as np

import layers

class DINet(nn.Module):

    def __init__(self, orig_model, arch, num_classes):
        super(DINet, self).__init__()

        self.arpool = layers.ApproximateRankPooling()
        self.l2norm = layers.L2Normalize()
        if arch.startswith('alexnet'):
            self.features = orig_model.features
            self.temppool = layers.TemporalPooling()
            self.classifier = nn.Sequential(nn.Dropout(),
                                            nn.Linear(256 * 6 * 6, 4096),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(4096, num_classes),
                                            nn.Softmax()
                                            )
            self.modelName = 'alexnet'
        else:
            raise NotImplementedError()

    def forward(self, x, vidids):
        # Approxmiate Rank Pooling layer : Get Dynamic Images
        dyn = self.arpool(x, vidids)
        # L2Normalize layer : Normalize the Dynamic Images
        params = [6e3, -128, 128, 0]
        params = Variable(torch.from_numpy(np.array(params)))
        nimgs = self.l2norm(dyn, params)
        # Initialize the result and intermediate tensors
        result = Variable(torch.FloatTensor(nimgs.size(0), 101))
        b = 0
        # Forward pass through Alexnet
        for batch in nimgs:
            # Debug code: Save and check dynamic images formed
            # fname = 'dynamic_image_' + str(b) + '.jpg'
            # save_image(tensor=batch.data, filename=fname)
            f = self.features(batch)
            t = self.temppool(f)
            t = t.view(t.size(0), 256 * 6 * 6)
            c = self.classifier(t)
            result[b, :] = c[0]
            b += 1
        return result
