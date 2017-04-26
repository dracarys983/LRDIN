import argparse
import os
import timeit
import time

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

    t_dataset_0 = timeit.default_timer()
    print "[INFO] Initializing the Dataset object"
    # Initialize the Dataset and Data Loader
    outdir = args.outdir
    UCF101 = data.UCF101(outdir, args.classfile)
    train_loader = data_utils.DataLoader(dataset=UCF101, batch_size=4,
        shuffle=False, num_workers=4)
    print "[INFO] Dataset object initialized"
    t_dataset_1 = timeit.default_timer()

    t_modelinit_0 = timeit.default_timer()
    # Initialize the Neural Network to be used
    print("[INFO] Using pre-trained model %s" % (args.arch))
    orig_model = tmodels.__dict__[args.arch](pretrained=True)
    dynamicImageNet = models.DINet(orig_model, args.arch, num_classes)
    print(dynamicImageNet)
    t_modelinit_1 = timeit.default_timer()

    t_load_0 = timeit.default_timer()
    i = 0
    start = time.time()
    for batch in train_loader:
        print("%i/%i, time=%.4f secs" % (i, len(UCF101), (time.time() - start)))
        i += batch[0].size(0)
        start = time.time()
    t_load_1 = timeit.default_timer()

    print("[TIME] Dataset Initialization: %.4f secs, Model Initialization: \
        %.4f secs, Batchwise loading: %.4f" % (t_dataset_1 - t_dataset_0,
        t_modelinit_1 - t_modelinit_0, t_load_1 - t_load_0))

if __name__ == '__main__':
    main()
