import argparse
import os
import shutil
import timeit
import time

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
import torchvision.models as tmodels
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.optim as optim

import models
import data

model_names = ('alexnet', 'resnet50', 'vgg16')

parser = argparse.ArgumentParser(description='PyTorch: UCF-101 Action Recognition')
parser.add_argument('--data', metavar='DIR', help='Path to Dataset')
parser.add_argument('--outdir', help='Path to output frame directory')
parser.add_argument('--traintest', type=str, help='Path to class train/test split files')
parser.add_argument('--classfile', type=str, help='Path to class IDs file')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    choices=model_names, help='model architecture: ' +
                    ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--classes', default=101, type=int,
                    help='Number of output classes for Action Recognition')
parser.add_argument('--print_debug', default=False, type=bool,
                    help='Print debug information')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


class AverageMeter(object):
    def __init__(self):
	self.reset()

    def reset(self):
	self.val = 0
	self.avg = 0
	self.sum = 0
	self.count = 0

    def update(self, val, n=1):
	self.val = val
	self.sum += val * n
	self.count += n
	self.avg = self.sum / self.count

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    #if is_best:
    #    shutil.copyfile(filename, 'model_best.pth.tar')

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    t_train_0 = timeit.default_timer()
    start = time.time()
    for i, (x, labels, vidids) in enumerate(train_loader):
	data_time.update(time.time() - start)

        targets = []
        for j in range(len(labels)):
            targets.append(labels[j][0]-1)
        targets = np.array(targets, dtype='int64')
        targets = Variable(torch.from_numpy(targets).cuda(async=False))
        vidids_var = Variable(vidids, requires_grad=False)
        x_var = Variable(x, requires_grad=False)

        y = model(x_var, vidids_var)
        loss = criterion(y, targets)

	prec1, prec5 = accuracy(y.data, targets.data, topk=(1, 5))
	losses.update(loss.data[0], x.size(0))
	top1.update(prec1[0], x.size(0))
	top5.update(prec5[0], x.size(0))

        optimizer.zero_grad()
        loss.backward(retain_variables=True)
        optimizer.step()

	batch_time.update(time.time() - start)
	start = time.time()

    	if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

        del y, loss, targets, x, x_var, vidids, vidids_var
    t_train_1 = timeit.default_timer()


def main():
    global args, model
    args = parser.parse_args()
    num_classes = args.classes

    t_dataset_0 = timeit.default_timer()
    print "[INFO] Initializing the Dataset object"
    # Initialize the Dataset and Data Loader
    outdir = args.outdir
    # Load train split 1
    UCF101 = data.UCF101(outdir, args.traintest, args.classfile)
    train_loader = data_utils.DataLoader(dataset=UCF101, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True)
    print "[INFO] Dataset object initialized"
    t_dataset_1 = timeit.default_timer()

    t_modelinit_0 = timeit.default_timer()
    # Initialize the Neural Network to be used
    print "[INFO] Using pre-trained model %s" % (args.arch)
    orig_model = tmodels.__dict__[args.arch](pretrained=True)
    dynamicImageNet = nn.DataParallel(models.DINet(orig_model, args.arch, num_classes), device_ids=[0]).cuda()
    print(dynamicImageNet)
    t_modelinit_1 = timeit.default_timer()

    criterion = nn.NLLLoss().cuda()
    optimizer = optim.SGD(dynamicImageNet.parameters(), lr=args.lr, momentum=args.momentum,
			weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            dynamicImageNet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, dynamicImageNet, criterion, optimizer, epoch)

        # evaluate on validation set
        #prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        #is_best = prec1 > best_prec1
        #best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': dynamicImageNet.state_dict(),
            #'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, #is_best
	)

if __name__ == '__main__':
    main()
