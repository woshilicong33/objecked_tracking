from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC
import cv2 as cv 


if __name__ == '__main__':
#X:\F_public\public\workspace\exchange\pedestrian\sampleclips\bsd\project_tracking\London\correct
    tracker = TrackerSiamFC(net_path=net_path)
    filePath = '/home/streamx/workspace/train4/'
#    '/home/streamx/workspace/tracking'
    for videoPath in os.listdir(filePath):
        if '.avi' in videoPath:
            tracker.track_video(filePath+videoPath)
