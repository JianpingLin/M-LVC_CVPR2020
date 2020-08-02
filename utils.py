import json
import sys
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import random
import os
import tensorflow as tf
from yuv_import import *
from msssim import MultiScaleSSIM as msssim_


def build_flows_pyramid(flows, num_levels):
    flows_pyramid = []
    flow_scales = [2.5, 5.0, 10., 20.]
    for l in range(num_levels):
        # Downsampling the scaled ground truth flow
        _, h, w, _ = tf.unstack(tf.shape(flows))
        downscale = 2 ** (num_levels - 1 - l)
        if l == num_levels - 1:
            flow_down = flows * flow_scales[l]
        else:
            flow_down = tf.image.resize_nearest_neighbor(flows,
                                                         (tf.floordiv(h, downscale), tf.floordiv(w, downscale))) * \
                        flow_scales[l]
        flows_pyramid.append(flow_down)

    return flows_pyramid

def get_patches(image_height, image_width):
    real_patch_heigt = 512
    real_patch_width = 512
    overlap = 64
    patch_heigt = real_patch_heigt + overlap
    patch_width = real_patch_width + overlap
    patches_num_height = image_height // real_patch_heigt
    patches_num_width = image_width // real_patch_width
    if image_height % real_patch_heigt > 0:
        patches_num_height += 1
    if image_width % real_patch_width > 0:
        patches_num_width += 1
    patches_h_list = []
    patches_w_list = []
    real_patches_h_list = []
    real_patches_w_list = []
    for patch_idx_h in range(patches_num_height):
        for patch_idx_w in range(patches_num_width):
            if patch_idx_h == 0:
                patches_h_start = 0
                real_patches_h_start = 0
                real_patches_h_end = real_patches_h_start + real_patch_heigt
            elif patch_idx_h == patches_num_height - 1:
                patches_h_start = image_height - patch_heigt
                real_patches_h_start = patch_heigt-(image_height - (patches_num_height-1)*real_patch_heigt)
                real_patches_h_end = patch_heigt
            else:
                patches_h_start = patch_idx_h * real_patch_heigt - overlap // 2
                real_patches_h_start = overlap // 2
                real_patches_h_end = real_patches_h_start + real_patch_heigt
            patches_h_end = patches_h_start + patch_heigt

            if patch_idx_w == 0:
                patches_w_start = 0
                real_patches_w_start = 0
                real_patches_w_end = real_patches_w_start + real_patch_width
            elif patch_idx_w == patches_num_width - 1:
                patches_w_start = image_width - patch_width
                real_patches_w_start = patch_width-(image_width - (patches_num_width-1)*real_patch_width)
                real_patches_w_end = patch_width
            else:
                patches_w_start = patch_idx_w * real_patch_width - overlap // 2
                real_patches_w_start = overlap // 2
                real_patches_w_end = real_patches_w_start + real_patch_width
            patches_w_end = patches_w_start + patch_width

            patches_h_list.append([patches_h_start, patches_h_end])
            patches_w_list.append([patches_w_start, patches_w_end])
            real_patches_h_list.append([real_patches_h_start, real_patches_h_end])
            real_patches_w_list.append([real_patches_w_start, real_patches_w_end])
    return patches_h_list,patches_w_list,real_patches_h_list,real_patches_w_list, patch_heigt, patch_width, patches_num_height, patches_num_width

def reshape2patches_tesnsor(tensor, patches_h_list, patches_w_list):
    patches_tensor_list = []
    for (patches_h, patches_w) in zip(patches_h_list, patches_w_list):
        patches_h_start, patches_h_end = patches_h
        patches_w_start, patches_w_end = patches_w
        patches_tensor_list.append(tensor[:,patches_h_start:patches_h_end,patches_w_start:patches_w_end,:])
    #patches_tensor = tf.stack(patches_tensor_list, axis=0)
    #patches_tensor = tf.squeeze(patches_tensor,axis=1)
    return patches_tensor_list

def reshape2image_tesnsor(patches_tensor_list, real_patches_h_list, real_patches_w_list, patches_num_height, patches_num_width):
    idx = 0
    patch_h_list = []
    for patch_idx_h in range(patches_num_height):
        patch_w_list = []
        for patch_idx_w in range(patches_num_width):
            real_patches_h_start, real_patches_h_end = real_patches_h_list[idx]
            real_patches_w_start, real_patches_w_end = real_patches_w_list[idx]
            patch = patches_tensor_list[idx][:, real_patches_h_start:real_patches_h_end, real_patches_w_start:real_patches_w_end, :]
            patch_w_list.append(patch)
            idx += 1
        patch_h = np.concatenate(tuple(patch_w_list),axis=2)
        patch_h_list.append(patch_h)
    tensor = np.concatenate(tuple(patch_h_list), axis=1)
    return tensor

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

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def get_test_class_dic(class_name):
    classes_dict = {
        'ClassD':
            {'sequence_name': ['RaceHorses', 'BQSquare', 'BlowingBubbles', 'BasketballPass'],
             'ori_yuv': ['RaceHorses_384x192_30.yuv', 'BQSquare_384x192_60.yuv',
                         'BlowingBubbles_384x192_50.yuv', 'BasketballPass_384x192_50.yuv'],
             'resolution': '384x192',
             'frameRate': [30, 60, 50, 50]
             },
        'ClassC':
            {'sequence_name': ['RaceHorsesC', 'BQMall', 'PartyScene', 'BasketballDrill'],
             'ori_yuv': ['RaceHorses_832x448_30.yuv', 'BQMall_832x448_60.yuv', 'PartyScene_832x448_50.yuv',
                         'BasketballDrill_832x448_50.yuv'],
             'resolution': '832x448',
             'frameRate': [30, 60, 50, 50]
             },
        'ClassE':
            {'sequence_name': ['FourPeople', 'Johnny', 'KristenAndSara'],
             'ori_yuv': ['FourPeople_1280x704_60.yuv', 'Johnny_1280x704_60.yuv',
                         'KristenAndSara_1280x704_60.yuv'],
             'resolution': '1280x704',
             'frameRate': [60, 60, 60]
             },
        'ClassB':
            {'sequence_name': ['Kimono', 'ParkScene', 'Cactus', 'BasketballDrive', 'BQTerrace'],
             'ori_yuv': ['Kimono_1920x1024_24.yuv', 'ParkScene_1920x1024_24.yuv', 'Cactus_1920x1024_50.yuv',
                         'BasketballDrive_1920x1024_50.yuv', 'BQTerrace_1920x1024_60.yuv'],
             'resolution': '1920x1024',
             'frameRate': [24, 24, 50, 50, 60]
             },
        'ClassUVG':
            {'sequence_name': ['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey',
                               'ReadySteadyGo', 'ShakeNDry', 'YachtRide'],
             'ori_yuv': ['Beauty_1920x1024_120fps.yuv', 'Bosphorus_1920x1024_120fps.yuv',
                         'HoneyBee_1920x1024_120fps.yuv', 'Jockey_1920x1024_120fps.yuv',
                         'ReadySteadyGo_1920x1024_120fps.yuv', 'ShakeNDry_1920x1024_120fps.yuv',
                         'YachtRide_1920x1024_120fps.yuv'],
             'resolution': '1920x1024',
             'frameRate': [120, 120, 120, 120, 120, 120, 120]
             }
    }

    return {class_name: classes_dict[class_name]}

def rgb_psnr_(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX**2 / mse)
def evaluate(img0, img1):
    img0 = img0.astype('float32')
    img1 = img1.astype('float32')
    rgb_psnr = rgb_psnr_(img0, img1)
    r_msssim = msssim_(img0[:, :, 0], img1[:, :, 0])
    g_msssim = msssim_(img0[:, :, 1], img1[:, :, 1])
    b_msssim = msssim_(img0[:, :, 2], img1[:, :, 2])
    rgb_msssim = (r_msssim + g_msssim + b_msssim)/3
    return rgb_psnr, rgb_msssim