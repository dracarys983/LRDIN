import torch.nn as nn

import layers

class DINet(nn.Module):

    def __init__(self, orig_model, arch, num_classes):
        super(DINet, self).__init__()

        self.arpool = nn.Sequential(layers.ApproximateRankPooling())
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
        f = self.features(dyn)
        y = self.classifier(f)
        return y
