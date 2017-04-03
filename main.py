import argparse
import os

import torch.nn as nn
import torch.utils.data
from torchvision.models import models

model_names = ('vgg16_bn', 'resnet50')

parser = argparse.ArgumentParser(description='PyTorch: UCF-101 Data Processing')
parser.add_argument('--data', metavar='DIR', help='Path to Dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names, help='model architecture: ' +
                    ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

def main():
    global args, model
    args = parser.parse_args()

    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arcg](pretrained=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    
