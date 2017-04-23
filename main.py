import argparse
import os

import torch.nn as nn
import torch.utils.data as data_utils
import torchvision.models as tmodels

import models
import data

model_names = ('alexnet', 'resnet50', 'vgg16')

parser = argparse.ArgumentParser(description='PyTorch: UCF-101 Action Recognition')
parser.add_argument('--data', metavar='DIR', help='Path to Dataset')
parser.add_argument('--outdir', help='Path to output frame directory')
parser.add_argument('--classfile', type=str, help='Path to class ID file')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    choices=model_names, help='model architecture: ' +
                    ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--classes', default=101, type=int,
                    help='Number of output classes for Action Recognition')

def main():
    global args, model
    args = parser.parse_args()
    num_classes = args.classes

    # Initialize the Dataset and Data Loader
    outdir = args.outdir
    UCF101 = data.UCF101(outdir, args.classfile)
    train_loader = data_utils.DataLoader(dataset=UCF101,
                                        batch_size=8,
                                        shuffle=False,
                                        num_workers=4)

    # Initialize the Neural Network to be used
    print("=> using pre-trained model '{}'".format(args.arch))
    orig_model = tmodels.__dict__[args.arch](pretrained=True)
    dynamicImageNet = DINet(orig_model, args.arch, num_classes)

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

    cudnn.benchmark = True


if __name__ == '__main__':
    main()
