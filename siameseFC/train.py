from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    root_dir = '/mnt/workspace/licong/Data/'
    # root_dir = '/mnt/ssd_disk_2/licong/Data/tracking'
    seqs = GOT10k(root_dir, subset='train', return_meta=True)
    tracker = TrackerSiamFC()
    tracker.train_over(seqs)
