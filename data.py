import os
import subprocess

def extract_frames(datadir, frames_per_second):
    subprocess.call(['extract_frames.sh', str(datadir), frames_per_second])

def approx_rankpool_kernel(datadir, dynimgdir, frames_per_dynimg):
    dirlist = os.listdir(datadir)
    for d in dirlist:
        framelist = os.listdir(d)
