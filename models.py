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
            self.classifier = nn.Sequential(nn.Dropout(),
                                            nn.Linear(256 * 6 * 6, 4096),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(4096, num_classes),
                                            )
            self.modelName = 'alexnet'
        else:
            raise NotImplementedError()

    def forward(self, x, vidids):
        dyn = self.arpool(x, vidids)
        result = Variable(torch.FloatTensor(dyn.size(0), dyn.size(1), 1, 101))
        inter = Variable(torch.FloatTensor(dyn.size(1), 1, 101))
        b = 0
        for batch in dyn:
            i = 0
            # Debug code: Save and check dynamic images formed
            # fname = 'dynamic_image_' + str(b) + '.jpg'
            # save_image(tensor=batch.data, filename=fname)
            for img in batch:
                # Send in dimensions 1 x C x H x W (for a single image)
                img = img.view(-1, img.size(0), img.size(1), img.size(2))
                f = self.features(img)
                f = f.view(f.size(0), 256 * 6 * 6)
                c = self.classifier(f)
                inter[i, :, :] = c
                i += 1
            result[b, :, :, :] = inter
            b += 1
        return result
