import torch.utils.data as data_utils
import torch

import os
import subprocess

import numpy as np
from PIL import Image

class UCF101(data_utils.Dataset):

    def __init__(self, datadir, classIdFile):
        self.datadir = datadir
        self.labels = os.listdir(datadir)

        # Construct the string to int mapping for labels
        f = open(classIdFile, 'r')
        classes = f.readlines()
        classes = [x.strip().split() for x in classes]
        self.intlabels = {}
        for x in classes:
            self.intlabels[x[1]] = int(x[0])

        # Construct the frames list for each video in each class
        # Get the total number of frames
        self.videolist = {}
        self.framemap = {}
        self.totalframes = 0
        for label in self.labels:
            vidpath = self.datadir + '/' + label
            vlist = [x for x in sorted(os.listdir(vidpath))]
            self.videolist[label] = vlist
            self.framemap[label] = {}
            for vid in vlist:
                fpath = vidpath + '/' + vid
                frames = [x for x in sorted(os.listdir(fpath))
                        if os.path.isfile(fpath + '/' + x)]
                self.totalframes += len(frames)
                self.framemap[label][vid] = frames

    def __getitem__(self, index):
        processed = 0
        global rlabel, rvid
        framenum = 0
        for label in self.labels:
            for vid in self.videolist[label]:
                for framelist in self.framemap[label][vid]:
                    if(processed + len(framelist) >= index):
                        framenum = index - processed
                        rlabel = label
                        rvid = vid
                        break
                    processed += len(framelist)
                    listnum += 1
        rlist = self.framemap[rlabel][rvid]
        rframe = rlist[framenum]
        filename = self.datadir + '/' + rlabel + '/' + rvid + '/' + rframe

        im = Image.open(filename)
        numpy_im = np.array([im])
        numpy_label = np.array([self.intlabels[rlabel]])
        return torch.from_numpy(numpy_im), torch.from_numpy(numpy_label)

    def __len__(self):
        return self.totalframes

def extract_frames(datadir, outdir, frames_per_second):
    inps = os.listdir(datadir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for inpdir in inps:
        inpdirpath = str(datadir) + '/' + str(inpdir)
        vids = os.listdir(inpdirpath)
        outdirpath = str(outdir) + '/' + str(inpdir)
        print inpdirpath, outdirpath
        if not os.path.exists(outdirpath):
            os.makedirs(outdirpath)
        for vid in vids:
            inpvidpath = str(inpdirpath) + '/' + str(vid)
            print inpvidpath
            subprocess.call(['./extract_frames.sh', inpvidpath, outdirpath, str(frames_per_second)])
