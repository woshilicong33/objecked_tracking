import cv2
import numpy as np
import caffe
import torch.nn.functional as F
import torch
def crop_and_resize(img, center, size, out_size,border_type=cv2.BORDER_CONSTANT,border_value=(0, 0, 0),interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)

    return patch
def init(box,img):
	# convert box to 0-indexed and center based [y, x, h, w]
	response_up = 16
	response_sz = 17
	scale_step = 1.0375
	scale_num = 3
	context = 0.5
	instance_sz = 255
	exemplar_sz = 127
	box = np.array([box[1] - 1 + (box[3] - 1) / 2,box[0] - 1 + (box[2] - 1) / 2,box[3], box[2]], dtype=np.float32)
	center, target_sz = box[:2], box[2:]

	print(center)
	#
	# # create hanning window
	upscale_sz = response_up * response_sz
	hann_window = np.outer(np.hanning(upscale_sz),np.hanning(upscale_sz))
	hann_window /= hann_window.sum()
	#
	# search scale factors
	scale_factors = scale_step ** np.linspace(-(scale_num // 2),scale_num // 2, scale_num)
	print(scale_factors)
	#
	# exemplar and search sizes
	context = context * np.sum(target_sz)
	z_sz = np.sqrt(np.prod(target_sz + context))
	x_sz = z_sz * instance_sz / exemplar_sz

	# exemplar image
	avg_color = np.mean(img, axis=(0, 1))

	z = crop_and_resize(
		img, center, z_sz,
		out_size = exemplar_sz,
		border_value = avg_color)
	#
	# # exemplar features
	# z = torch.from_numpy(z).to(
	# 	self.device).permute(2, 0, 1).unsqueeze(0).float()
	# self.kernel = self.net.backbone(z)
	return z, z_sz, scale_factors, avg_color, x_sz,center, hann_window, target_sz
img_s = cv2.imread('./image/00000001.jpg')
img_x = cv2.imread('./image/00000120.jpg')
protofile_127 = "./AlexNetV1_127.prototxt"
modelFile_127 = "./AlexNetV1_127.caffemodel"
net_127 = caffe.Net(protofile_127, modelFile_127, caffe.TEST)

protofile_255 = "./AlexNetV1_255.prototxt"
modelFile_255 = "./AlexNetV1_255.caffemodel"
net_255 = caffe.Net(protofile_255, modelFile_255, caffe.TEST)

box = open('./image/groundtruth.txt','r').readlines()
x,y,w,h = float(box[0].split(',')[0]),float(box[0].split(',')[1]),float(box[0].split(',')[2]),float(box[0].split(',')[3])
box = [x,y,w,h]

cap = cv2.VideoCapture('./image/testVideo.avi')
cnt = 0
while (cap.isOpened()):
	ret, frame = cap.read()
	showimage = frame
	box_input = cv2.selectROI(windowName="roi", img=frame, showCrosshair=True, fromCenter=False)
	cnt +=1
	if cnt == 1:
		z, z_sz, scale_factors, avg_color, x_sz, center, hann_window, target_sz = init(box_input, frame)

		img = z.astype(np.float32)
		img = img.transpose((2, 0, 1))
		net_127.blobs["data"].data[...] = img
		out_127 = net_127.forward()
	else:
		x = [crop_and_resize(frame, center, x_sz * f, out_size=255, border_value=avg_color) for f in scale_factors]
		x = np.stack(x, axis=0)
		x = x.astype(np.float32)
		x = x.transpose((0, 3, 1, 2))

		net_255.blobs["data"].data[...] = x
		out_255 = net_255.forward()

		nz = np.shape(out_127['conv5'])[0]
		nx, c, h, w = np.shape(out_255['conv5'])
		# out_255['conv5'] = out_255['conv5'].view(-1, nz * c, h, w)
		out_255['conv5'] = np.reshape(out_255['conv5'],(3, 256, 22, 22))
		print('out_127:',np.shape(out_127['conv5']))
		print('out_255:',np.shape(out_255['conv5']))
		x = torch.from_numpy(out_255['conv5'])
		z = torch.from_numpy(out_127['conv5'])

		out = F.conv2d(x, z, groups=nz)

		responses = out.view(nx, -1, out.size(-2), out.size(-1)) * 0.001
		responses = responses.squeeze(1).cpu().numpy()
		responses = np.stack([cv2.resize(u, (16*17, 16*17),interpolation=cv2.INTER_CUBIC)for u in responses])
		responses[:3 // 2] *= 0.9745
		responses[3 // 2 + 1:] *= 0.9745

		# peak scale
		scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

		# peak location
		response = responses[scale_id]
		response -= response.min()
		response /= response.sum() + 1e-16

		response = (1 - 0.176) * response + 0.176 * hann_window

		loc = np.unravel_index(response.argmax(), response.shape)

		# locate target center
		disp_in_response = np.array(loc) - (16*17 - 1) / 2
		disp_in_instance = disp_in_response * 8 / 16
		disp_in_image = disp_in_instance * x_sz * scale_factors[scale_id] / 255
		center += disp_in_image

		# update target size
		scale = (1 - 0.59) * 1.0 + 0.59 * scale_factors[scale_id]
		target_sz *= scale
		z_sz *= scale
		x_sz *= scale
		# return 1-indexed and left-top based bounding box
		box = np.array([
			center[1] + 1 - (target_sz[1] - 1) / 2,
			center[0] + 1 - (target_sz[0] - 1) / 2,
			target_sz[1], target_sz[0]])
		print(box)
		cv2.rectangle(showimage,(int(box[0]),int(box[1])),(int(box[0]+box[2]),int(box[1]+box[3])),(0,255,255),4,1)
		cv2.imshow("showImage",showimage)
		cv2.waitKey(30)

cap.release()

