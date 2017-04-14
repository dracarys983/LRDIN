import torch.nn as nn

class ApproximateRankPooling(nn.Module):

    def __init__(self, D_in, D_out):
        super(ApproximateRankPooling, self).__init__()
