import torch.utils.data as data_utils
import torch

import os
import subprocess
import math

import numpy as np
from PIL import Image

class UCF101(data_utils.Dataset):

    def __init__(self, datadir, classIdFile, train=True):
        self.datadir = datadir
        self.labels = os.listdir(datadir)
        self.train = train

        # Construct the string to int mapping for labels
        f = open(classIdFile, 'r')
        classes = f.readlines()
        classes = [x.strip().split() for x in classes]
        self.intlabels = {}
        for x in classes:
            self.intlabels[x[1]] = int(x[0])

        # Construct the frames list for each video in each class
        # Get the total number of videos (batch size refers to videos)
        self.videolist = {}
        self.framemap = {}
        self.totalvids = 0
        for label in self.labels:
            vidpath = self.datadir + '/' + label
            vlist = [x for x in sorted(os.listdir(vidpath))]
            self.videolist[label] = vlist
            self.totalvids += len(vlist)
            self.framemap[label] = {}
            for vid in vlist:
                fpath = vidpath + '/' + vid
                frames = [x for x in sorted(os.listdir(fpath))
                        if os.path.isfile(fpath + '/' + x)]
                self.framemap[label][vid] = frames

    def __getitem__(self, index):
        train_data = []
        train_labels = []
        processed = 0
        for label in self.labels:
            flag = 0
            for vidi in range(len(self.videolist[label])):
                if processed+vidi == index:
                    vidname = self.videolist[label][vidi]
                    framelist = self.framemap[label][vidname]
                    stepSize = 6
                    dynFrames = 10      # Number of frames per Dynamic Image
                    dynImages = 10      # Number of Dymamic Images to max pool
                    nFrames = len(framelist)
                    nSteps = 0
                    for i in range(0, nFrames, stepSize):
                        nSteps += 1
                    if nSteps > 1 and nSteps > dynFrames:
                       dynImages = min(dynImages, int(math.ceil(0.75 * nSteps)))
                        rpermi = np.random.permutation(nSteps)
                        rpermi = rpermi[:dynImages]
                        rselect = [0] * nSteps
                        rselect = [1 if x in rpermi else rselect[x] for x in range(len(rselect))]
                    else:
                        rselect = [1] * nSteps
                    count = 0
                    vidid = 0
                    resz = (227, 227)
                    ims = []
                    labs = []
                    self.vidids = []
                    for i in range(0, nFrames, stepSize):
                        if rselect[count]:
                            idx = [x for x in range(i, min(i+dynFrames-1, nFrames))]
                            self.vidids.extend([vidid] * len(idx))
                            vidid += 1
                            for ind in idx:
                                frame = framelist[ind]
                                fpath = self.datadir + '/' + label + '/' + vidname + '/' + frame
                                img = Image.open(fpath)
                                img = img.resize(resz, Image.ANTIALIAS)
                                im = np.array(img)
                                ims.append(im)
                                labs.append(self.intlabels[label])
                        count += 1
                    train_data.extend(np.array(ims, dtype='float32'))
                    train_labels.extend(np.array(labs, dtype='int32'))
                    flag = 1
                    break
            processed += len(self.videolist[label])
            if flag:
                break
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        s1 = train_data.shape[0]
        s2 = train_labels.shape[0]
        diff = int(90 - s1)
        if diff:
            z = np.zeros(np.insert(np.array(train_data.shape[1:]), 0, diff, axis=0), dtype='float32')
            train_data = np.concatenate((train_data, z))
            z = np.zeros(np.array(diff,), dtype='int32')
            train_labels = np.concatenate((train_labels, z))
        return torch.from_numpy(np.array(train_data)), torch.from_numpy(np.array(train_labels))

    def __len__(self):
        return self.totalvids

def extract_frames(datadir, outdir, frames_per_second):
    inps = os.listdir(datadir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for inpdir in inps:
        inpdirpath = str(datadir) + '/' + str(inpdir)
        vids = os.listdir(inpdirpath)
        outdirpath = str(outdir) + '/' + str(inpdir)
        if not os.path.exists(outdirpath):
            os.makedirs(outdirpath)
        for vid in vids:
            inpvidpath = str(inpdirpath) + '/' + str(vid)
            subprocess.call(['./extract_frames.sh', inpvidpath, outdirpath, str(frames_per_second)])
