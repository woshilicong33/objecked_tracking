import numpy as np
import torch
import torch.nn as nn
import cv2
import sys
sys.path.append("..")
caffe_root = '/home/streamx/workspace/caffe-base/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

protofile = "./AlexNetV1.prototxt"
modelFile = "./AlexNetV1.caffemodel"
modelPath = "../pretrained/siamfc_alexnet_e1.pth"

net = caffe.Net(protofile, modelFile, caffe.TEST)
img = cv2.imread("test.jpg")
img = cv2.resize(img,(255,255))
img = img.astype(np.float32)
img = img.transpose((2,0,1))
net.blobs["data"].data[...] = img
out = net.forward()




network = torch.load(modelPath)
print(network)




