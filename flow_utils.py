import numpy as np
import matplotlib

matplotlib.use('Agg')
from pylab import box
import matplotlib.pyplot as plt
# import cv2
import sys
import argparse
import math

__all__ = ['load_flow', 'save_flow', 'vis_flow']


def load_flow(path):
    with open(path, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count=1)[0])
        if magic == 202021.25:
            w, h = np.fromfile(f, np.int32, count=1)[0], np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=h * w * 2)
            data.resize((h, w, 2))
            return data
        return None


def save_flow(path, flow):
    magic = np.array([202021.25], np.float32)
    h, w = flow.shape[:2]
    h, w = np.array([h], np.int32), np.array([w], np.int32)

    with open(path, 'wb') as f:
        magic.tofile(f);
        w.tofile(f);
        h.tofile(f);
        flow.tofile(f)


def makeColorwheel():
    #  color encoding scheme

    #   adapted from the color circle idea described at
    #   http://members.shaw.ca/quadibloc/other/colint.htm

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])  # r g b

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
    col += RY

    # YG
    colorwheel[col:YG + col, 0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
    colorwheel[col:YG + col, 1] = 255;
    col += YG;

    # GC
    colorwheel[col:GC + col, 1] = 255
    colorwheel[col:GC + col, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
    col += GC;

    # CB
    colorwheel[col:CB + col, 1] = 255 - np.floor(255 * np.arange(0, CB, 1) / CB)
    colorwheel[col:CB + col, 2] = 255
    col += CB;

    # BM
    colorwheel[col:BM + col, 2] = 255
    colorwheel[col:BM + col, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
    col += BM;

    # MR
    colorwheel[col:MR + col, 2] = 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
    colorwheel[col:MR + col, 0] = 255
    return colorwheel


def computeColor(u, v):
    colorwheel = makeColorwheel();
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v)

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)  # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)  # 1, 2, ..., ncols
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1], 3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])  # increase saturation with radius
        col[~idx] *= 0.75  # out of range
        img[:, :, 2 - i] = np.floor(255 * col).astype(np.uint8)

    return img.astype(np.uint8)


def vis_flow(flow):
    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999

    maxrad = -1
    # fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0
    v[greater_v] = 0

    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])

    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    maxrad = max([maxrad, np.amax(rad)])
    # print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

    u = u / (maxrad + eps)
    v = v / (maxrad + eps)
    img = computeColor(u, v)
    return img[:, :, [2, 1, 0]]

def vis_flow_image_final(flow_pyramid, flow_gt_pyramid, images_list,gray_images_list, filename='./flow.png'):
    num_contents = len(flow_pyramid) + len(flow_gt_pyramid) + len(images_list)+ len(gray_images_list)
    nums_list = [len(flow_pyramid), len(flow_gt_pyramid), len(images_list), len(gray_images_list)]
    nums_list.sort()
    cols = nums_list[-2]
    if cols <= 3:
        cols = nums_list[-1]
    cols=4
    rows = math.ceil(num_contents / cols)

    fig_dpi=200
    plt.rcParams['savefig.dpi'] = fig_dpi
    plt.rcParams['figure.dpi'] = fig_dpi

    fig = plt.figure()

    fig_id = 1

    for image in images_list:
        plt.subplot(rows, cols, fig_id)
        plt.imshow(image)
        plt.tick_params(labelbottom=False, bottom=False)
        plt.tick_params(labelleft=False, left=False)
        plt.xticks([])
        box(False)
        fig_id += 1

    for image in gray_images_list:
        plt.subplot(rows, cols, fig_id)
        plt.imshow(image,cmap='gray')
        plt.tick_params(labelbottom=False, bottom=False)
        plt.tick_params(labelleft=False, left=False)
        plt.xticks([])
        box(False)
        fig_id += 1

    for flow_gt in flow_gt_pyramid:
        plt.subplot(rows, cols, fig_id)
        plt.imshow(vis_flow(flow_gt))
        plt.tick_params(labelbottom=False, bottom=False)
        plt.tick_params(labelleft=False, left=False)
        plt.xticks([])
        box(False)

        fig_id += 1
    for flow in flow_pyramid:
        plt.subplot(rows, cols, fig_id)
        plt.imshow(vis_flow(flow))
        plt.tick_params(labelbottom=False, bottom=False)
        plt.tick_params(labelleft=False, left=False)
        plt.xticks([])
        box(False)

        fig_id += 1

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=fig_dpi)
    plt.close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    flow = load_flow('13382_flow.flo')
    flow = load_flow('datasets/Sintel/training/flow/alley_1/frame_0001.flo')
    img = vis_flow(flow)
    import imageio

    imageio.imsave('test.png', img)
    # import cv2
    # cv2.imshow('', img[:,:,:])
    # cv2.waitKey()
