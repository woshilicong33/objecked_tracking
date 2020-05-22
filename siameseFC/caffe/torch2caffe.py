import sys
from collections import OrderedDict
sys.path.append("..")
# from backbones import AlexNetV1
import numpy as np
import torch
import torch.nn as nn
caffe_root = '/home/streamx/workspace/caffe-base/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
protofile = "./AlexNetV1.prototxt"
modelFile = "./AlexNetV1.caffemodel"
modelPath = "../pretrained/siamfc_alexnet_e1.pth"
net = caffe.Net(protofile, caffe.TEST)
caffeParams = net.params
for k in sorted(caffeParams):
    print(k)
print(len(caffeParams))

network = torch.load(modelPath)
for param_tensor, value in network.state_dict().items():
    print(param_tensor)
    # print(value.shape)
    # if "bias" in param_tensor:
    #     print(value)
print((len(network.state_dict())))
recycle = 0
layerNum = 1
i = 0
sizeNum = 0
## pnet
nameDict = {
            "backbone.conv1.0.weight":"conv1,0",
            "backbone.conv1.0.bias":"conv1,1",
            "backbone.conv1.1.weight":"conv1/scale,0",
            "backbone.conv1.1.bias":"conv1/scale,1",
            "backbone.conv1.1.running_mean":"conv1/bn,0",
            "backbone.conv1.1.running_var":"conv1/bn,1",
            "backbone.conv1.1.num_batches_tracked":"conv1/bn,2",

            "backbone.conv2.0.weight":"conv2,0",
            "backbone.conv2.0.bias":"conv2,1",
            "backbone.conv2.1.weight":"conv2/scale,0",
            "backbone.conv2.1.bias":"conv2/scale,1",
            "backbone.conv2.1.running_mean":"conv2/bn,0",
            "backbone.conv2.1.running_var":"conv2/bn,1",
            "backbone.conv2.1.num_batches_tracked":"conv2/bn,2",

            "backbone.conv3.0.weight":"conv3,0",
            "backbone.conv3.0.bias":"conv3,1",
            "backbone.conv3.1.weight":"conv3/scale,0",
            "backbone.conv3.1.bias":"conv3/scale,1",
            "backbone.conv3.1.running_mean":"conv3/bn,0",
            "backbone.conv3.1.running_var":"conv3/bn,1",
            "backbone.conv3.1.num_batches_tracked":"conv3/bn,2",

            "backbone.conv4.0.weight":"conv4,0",
            "backbone.conv4.0.bias":"conv4,1",
            "backbone.conv4.1.weight":"conv4/scale,0",
            "backbone.conv4.1.bias":"conv4/scale,1",
            "backbone.conv4.1.running_mean":"conv4/bn,0",
            "backbone.conv4.1.running_var":"conv4/bn,1",
            "backbone.conv4.1.num_batches_tracked":"conv4/bn,2",

            "backbone.conv5.0.weight":"conv5,0",
            "backbone.conv5.0.bias":"conv5,1",
}
pytorchLayerNameList = list(nameDict.keys())
caffeLayerNameList = list(nameDict.values())
for param_tensor in network.state_dict():
    print(str(i)+' |', param_tensor+' |', nameDict[param_tensor])
    if param_tensor not in pytorchLayerNameList:
        print("there is some problem in nameDict")
        sys.exit()

    param = network.state_dict()[param_tensor]
    # print('param.cpu().data.numpy():', param.cpu().data.numpy().shape)

    caffeLayerPara = nameDict[param_tensor]
    # print('caffeLayerPara:', caffeLayerPara)

    if "," in caffeLayerPara:
        caffeLayerName, caffeLayerMatNum = caffeLayerPara.strip().split(",")
        caffeLayerMatNum = int(caffeLayerMatNum)
        if caffeLayerName not in caffeParams:
            print("caffeLayerName is not in caffe")
        print('param.cpu().data.numpy():', param.cpu().data.numpy().shape)
        # print(caffeParams[caffeLayerName][caffeLayerMatNum].data[...])
        print('caffe layer shape:', caffeParams[caffeLayerName][caffeLayerMatNum].data[...].shape)
        print('==================================')
        if "num_batches_tracked" in param_tensor:
            caffeParams[caffeLayerName][caffeLayerMatNum].data[...] = np.array([1.0])
        else:
            caffeParams[caffeLayerName][caffeLayerMatNum].data[...] = param.cpu().data.numpy()

    i += 1

net.save(modelFile) 
print("net save end")