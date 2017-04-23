import torch.nn as nn

import layers

class DINet(nn.Module):

    def __init__(self, orig_model, arch, num_classes):
        super(DINet, self).__init__()

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

    def forward(self, x, nvids, dframes, vidids):
        dyn = layers.ApproximateRankPooling()(x, nvids, dframes, vidids)
        f = self.features(dyn)
        y = self.classifier(f)
        return y
