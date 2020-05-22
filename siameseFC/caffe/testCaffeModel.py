import numpy as np
import torch
import torch.nn as nn
import sys
from PIL import Image
sys.path.append("..")
import torchvision.transforms as transforms
# caffe_root = '/home/streamx/workspace/caffe-base/'
# sys.path.insert(0, caffe_root + 'python')
import caffe  
data_transform = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])
protofile = "./AlexNetV1.prototxt"
modelFile = "./AlexNetV1.caffemodel"
modelPath = "../pretrained/siamfc_alexnet_e1.pth"

net = caffe.Net(protofile, modelFile, caffe.TEST)

img=Image.open("test.jpg")
img = img.resize((127,127))
img_tensor = data_transform(img).unsqueeze(0)
img_tensor = img_tensor.to(torch.device('cuda'))

network = torch.load(modelPath)
network.eval()
out_pytroch = network.forward(img_tensor,img_tensor)

net.blobs["data"].data[...] = img_tensor.cpu().numpy()
out_caffe = net.forward()

print('out_caffe:',out_caffe['conv5'])



