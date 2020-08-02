import numpy as np
import os
import cv2
import math
# import random
# import ipdb
MSB2Num = [0,256,512,768] # for 10bit yuv, the Higher 2 bit could only be 00,01,10,11. The pixel_value = LSB_value + MSB2Num[MSB_value]
## randomly extract few frames from a sequence
def yuv420_import(filename,height,width,Org_frm_list,extract_frm,flag,isRef,isList0,deltaPOC,isLR):
    # print filename, height, width
    if flag:
        frm_size = int(float(height*width*3)/float(2*2)) ## for 10bit yuv, each pixel occupy 2 byte
    else:
        frm_size = int(float(height*width*3)/2)
    if isLR:
        frm_size=frm_size*4
        if flag:
            row_size = width * 2
        else:
            row_size = width
    Luma = []
    U=[]
    V=[]
    # Org_frm_list = range(1,numfrm)
    # random.shuffle(Org_frm_list)
    with open(filename,'rb') as fd:
        for extract_index in range(extract_frm):
            if isRef:
                if isList0:
                    current_frm = Org_frm_list[extract_index]-deltaPOC
                else:
                    current_frm = Org_frm_list[extract_index]+deltaPOC
            else:
                current_frm = Org_frm_list[extract_index]
            fd.seek(frm_size*current_frm,0)
            # ipdb.set_trace()
            if flag:
                Yt = np.zeros((height,width),np.uint16,'C')
                for m in range(height):
                    for n in range(width):
                        symbol = fd.read(2)
                        LSB = ord(symbol[0])
                        MSB = ord(symbol[1])
                        Pixel_Value = LSB+MSB2Num[MSB]
                        Yt[m,n]=Pixel_Value
                    if isLR:
                        fd.seek(row_size, 1)
                Luma.append(Yt)
                del Yt
            else:
                Yt = np.zeros((height,width),np.uint8,'C')
                for m in range(height):
                    for n in range(width):
                        symbol = fd.read(1)
                        Pixel_Value = ord(symbol)
                        Yt[m,n]=Pixel_Value
                    if isLR:
                        fd.seek(row_size, 1)
                Luma.append(Yt)
                del Yt
                Ut = np.zeros((height//2, width//2), np.uint8, 'C')
                for m in range(height//2):
                    for n in range(width//2):
                        symbol = fd.read(1)
                        Pixel_Value = ord(symbol)
                        Ut[m, n] = Pixel_Value
                    if isLR:
                        fd.seek(row_size, 1)
                U.append(Ut)
                del Ut
                Vt = np.zeros((height // 2, width // 2), np.uint8, 'C')
                for m in range(height // 2):
                    for n in range(width // 2):
                        symbol = fd.read(1)
                        Pixel_Value = ord(symbol)
                        Vt[m, n] = Pixel_Value
                    if isLR:
                        fd.seek(row_size, 1)
                V.append(Vt)
                del Vt
    return Luma,U,V

def psnr(target, ref):
    target_data = np.asarray(target, 'f')
    ref_data = np.asarray(ref, 'f')
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = np.mean(diff ** 2.)
    return 10 * math.log10(255 ** 2. / rmse)

def YUV2RGB420_custom(Y_frames,U_frames,V_frames):
	shape = np.shape(Y_frames)
	n = shape[0]
	h = shape[1]
	w = shape[2]
	RGB_frames = np.zeros([n, h, w, 3], np.uint8)
	yuv_frame=np.zeros([h*3//2,w],np.uint8)

	for n_i in range(n):
		for f_i in range(1):
			Y = Y_frames[n_i, :, :, 0]
			U = U_frames[n_i, :, :, 0]
			V = V_frames[n_i, :, :, 0]
			yuv_frame[:h,:]=Y
			yuv_frame[h:5 * h // 4, :] = np.reshape(U, [h // 4, w])
			yuv_frame[5 * h // 4 :, :] = np.reshape(V, [h // 4, w])
			rgb = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB_I420)
			RGB_frames[n_i,:,:,:]=rgb
	return RGB_frames