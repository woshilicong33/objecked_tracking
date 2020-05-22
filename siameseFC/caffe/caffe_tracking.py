import cv2
import numpy as np
def init(image):

	return 0
img_s = cv2.imread('./image/00000001.jpg')
img_x = cv2.imread('./image/000000090.jpg')

box = open('./image/groundtruth.txt','r').readlines()
x,y,w,h = box[0].split(',')[0],box[0].split(',')[1],box[0].split(',')[2],box[0].split(',')[3]

