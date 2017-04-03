import subprocess

def extract_frames(datadir, frames_per_second):
    subprocess.call(['extract_frames.sh', str(datadir), frames_per_second])

def
