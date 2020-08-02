import os
import argparse
import time
import tensorflow as tf
import numpy as np
import cv2
from model import *
from utils import *
from flow_utils import vis_flow_image_final
from yuv_import import *
import flownet_models as models
from PIL import Image


class Tester(object):
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        value_class = get_test_class_dic(self.args.test_class)[self.args.test_class]
        size_list = value_class['resolution'].split('x')
        width = int(size_list[0])
        height = int(size_list[1])
        self.image_size = [height, width]
        if not ((self.image_size[0] > 512) and (self.image_size[1] > 512)):
            self.patch_height, self.patch_width = self.image_size
        else:
            self.patches_h_list, self.patches_w_list, self.real_patches_h_list, self.real_patches_w_list, self.patch_height, self.patch_width, self.patch_num_heigt, self.patch_num_width = get_patches(
                self.image_size[0], self.image_size[1])
        self.Chroma_size = [size // 2 for size in self.image_size]
        self._build_graph()

    def _build_graph(self):
        # Input images and ground truth optical flow definition
        with tf.name_scope('Data'):
            self.images_pre_rec_multiframes = tf.placeholder(tf.float32,
                                                             shape=(self.args.batch_size, *self.image_size, 3, 4),
                                                             name='images_pre_rec_multiframes')
            self.images_pre_rec_4, self.images_pre_rec_3, self.images_pre_rec_2, self.images_pre_rec = tf.unstack(
                self.images_pre_rec_multiframes, axis=4)
            self.images_cur_ori = tf.placeholder(tf.float32, shape=(self.args.batch_size, *self.image_size, 3),
                                                 name='images_cur_ori')
            self.images_cur_ori_patch = tf.placeholder(tf.float32, shape=(
            self.args.batch_size, self.patch_height, self.patch_width, 3),
                                                       name='images_cur_ori_patch')
            self.images_pre_rec_patch = tf.placeholder(tf.float32, shape=(
            self.args.batch_size, self.patch_height, self.patch_width, 3),
                                                       name='.images_pre_rec_patch')
            self.flow_ori_input = tf.placeholder(tf.float32, shape=(self.args.batch_size, *self.image_size, 2),
                                                 name='flow_ori_input')
            self.flow_rec_input = tf.placeholder(tf.float32, shape=(self.args.batch_size, *self.image_size, 2),
                                                 name='flow_rec_input')
            self.images_cur_pred_input = tf.placeholder(tf.float32, shape=(self.args.batch_size, *self.image_size, 3),
                                                        name='images_cur_pred_input')
            self.flows_pre_rec = tf.placeholder(tf.float32, shape=(self.args.batch_size, *self.image_size, 2, 3),
                                                name='flows_pre_rec')
            self.flow_3_pre_rec, self.flow_2_pre_rec, self.flow_1_pre_rec = tf.unstack(self.flows_pre_rec, axis=4)
        ########ME-Net########
        memodel = models.FlowNet2(height=self.patch_height, width=self.patch_width, name='flownet2')
        self.flow_ori_patch = tf.transpose(memodel.build_graph(tf.transpose(self.images_cur_ori_patch, [0, 3, 1, 2]),
                                                               tf.transpose(self.images_pre_rec_patch, [0, 3, 1, 2])),[0, 2, 3, 1]) / 20.0

        ########MAMVP-Net########
        flows_1_pre_rec_pyramid = build_flows_pyramid(self.flow_1_pre_rec, self.args.num_levels)
        flows_2_pre_rec_pyramid = build_flows_pyramid(self.flow_2_pre_rec, self.args.num_levels)
        flows_3_pre_rec_pyramid = build_flows_pyramid(self.flow_3_pre_rec, self.args.num_levels)
        mamvpmodel = MAMVPNet(num_levels=self.args.num_levels,
                              warp_type=self.args.warp_type,
                              use_dc=self.args.use_dc,
                              output_level=self.args.output_level,
                              name='amvpnet')
        self.flow_pred, _, _ = mamvpmodel(flows_3_pre_rec_pyramid, flows_2_pre_rec_pyramid,
                                          flows_1_pre_rec_pyramid)

        self.flow_diff = self.flow_ori_input - self.flow_pred
        ########MVD Autoencoder########
        num_pixels = self.args.batch_size * self.image_size[0] * self.image_size[1]
        mvdmodel = bls2017ImgCompression_mvd_factor(2, self.args.mvd_M_filters, name='mvdnet')
        self.bit_string_mvd, entropy_bottleneck_mvd, self.flow_diff_rec_0, mvd_train_bpp = mvdmodel(
            self.flow_diff, num_pixels, reuse=False, isTrain=False)

        flow_rec = self.flow_pred + self.flow_diff_rec_0
        self.flow_rec_0 = flow_rec

        ########MV Refine-Net########
        mvlfmodel = MVLoopFiltering(name='mvlfmodel')
        self.flow_rec = mvlfmodel(self.flow_3_pre_rec, self.flow_2_pre_rec, self.flow_1_pre_rec, flow_rec,
                                  self.images_pre_rec)
        self.flow_diff_rec = self.flow_rec - self.flow_pred

        ########MMC-Net########
        mcmodel = MCNet_Multiple(name='MCNet')
        self.images_cur_pred, features_warped = mcmodel(self.images_pre_rec_4, self.images_pre_rec_3,
                                                        self.images_pre_rec_2,
                                                        self.images_pre_rec, self.flow_3_pre_rec, self.flow_2_pre_rec,
                                                        self.flow_1_pre_rec, self.flow_rec_input)

        self.images_cur_resi = self.images_cur_ori - self.images_cur_pred_input
        ########Residual Autoencoder########
        resimodel = bls2017ImgCompression_resi_RGB(3, self.args.resi_N_filters, self.args.resi_M_filters,
                                                   name='resinet')
        self.bit_string_resi, entropy_bottleneck_resi, tensor_tilde, images_cur_resi_train_bpp, self.bit_string_resi_dev, entropy_bottleneck_dev, _, resi_dev_train_bpp = resimodel(
            self.images_cur_resi, num_pixels, reuse=False, isTrain=False)
        ########Residual Refine-Net########
        resideblurmodel = ResiDeBlurNet(name='resideblurmodel')
        self.images_cur_resi_rec = resideblurmodel(tensor_tilde, self.images_cur_pred, features_warped)
        self.images_cur_rec = self.images_cur_pred_input + self.images_cur_resi_rec

        model_vars_restore = memodel.vars + mamvpmodel.vars + mvdmodel.vars + mvlfmodel.vars + mcmodel.vars + resimodel.vars + resideblurmodel.vars

        with tf.name_scope('Loss'):
            self._losses_mvd = []
            self._losses_resi = []
            loss = tf.reduce_mean(tf.squared_difference(self.images_cur_ori * 255.0, self.images_cur_rec * 255.0))
            self._losses_resi.append(loss)
            self._losses_mvd.append(mvd_train_bpp)
            self._losses_resi.append(images_cur_resi_train_bpp)
            self._losses_resi.append(resi_dev_train_bpp)

        # Initialization
        self.sess.run(tf.global_variables_initializer())

        if self.args.resume is not None:
            saver_0 = tf.train.Saver(model_vars_restore)
            print(f'Loading learned model from checkpoint {self.args.resume}')
            saver_0.restore(self.sess, self.args.resume)

        componet = ['RGB']
        PSNR_sum_list = []
        self._PSNR_list = []
        self._MSSSIM_list = []
        for i in range(len(componet)):
            ori = self.images_cur_ori[i] * 255
            rec = tf.round(tf.clip_by_value(self.images_cur_rec[i], 0, 1) * 255)
            PSNR = tf.squeeze(tf.image.psnr(ori, rec, 255))
            MSSSIM = tf.squeeze(tf.image.ssim_multiscale(ori, rec, 255))
            self._PSNR_list.append(PSNR)
            self._MSSSIM_list.append(MSSSIM)
            PSNR_sum_list.append(tf.summary.scalar('PSNR/' + componet[i], PSNR))

    def test(self):
        Orig_dir = self.args.test_seq_dir
        x265enc_dir = os.path.join(self.args.exp_data_dir, 'I_frames_enc')
        # crf_list=[15,19,23,27,31,35,39,43]
        qp_dic = {16: 21, 24: 23, 40: 25, 64: 27}
        qp_list = [qp_dic[self.args.lmbda]]
        frames_to_be_encoded = 100
        Org_frm_list = list(range(frames_to_be_encoded))
        classes_dict = get_test_class_dic(self.args.test_class)
        for key_class, value_class in classes_dict.items():
            size_list = value_class['resolution'].split('x')
            width = int(size_list[0])
            height = int(size_list[1])
            for seq_idx in range(len(value_class['sequence_name'])):
                for qp in qp_list:
                    ori_filename = os.path.join(Orig_dir, value_class['ori_yuv'][seq_idx])
                    print(key_class, value_class['sequence_name'][seq_idx], 'qp' + str(qp))
                    bits_list = []
                    RGB_PSNR_list = []
                    RGB_MSSSIM_list = []

                    ori_all_Y_list, ori_all_U_list, ori_all_V_list = yuv420_import(ori_filename, height, width,
                                                                                   Org_frm_list, len(Org_frm_list),
                                                                                   False, False, False, 0, False)
                    ori_Y = ori_all_Y_list[0][np.newaxis, :, :, np.newaxis]
                    ori_U = ori_all_U_list[0][np.newaxis, :, :, np.newaxis]
                    ori_V = ori_all_V_list[0][np.newaxis, :, :, np.newaxis]
                    RGB_ori = np.squeeze(YUV2RGB420_custom(ori_Y, ori_U, ori_V))

                    ori_file = os.path.join(x265enc_dir, value_class['sequence_name'][seq_idx] + '_' + value_class[
                        'resolution'] + '.png')
                    img_ori_save = Image.fromarray(RGB_ori)
                    img_ori_save.save(ori_file)
                    bin_file = os.path.join(x265enc_dir, 'enc_' + value_class['sequence_name'][seq_idx] + '_' + value_class[
                                                'resolution'] + '_' + str(value_class['frameRate'][seq_idx]) + '_qp' + str(qp) + '.bpg')
                    rec_file = os.path.join(x265enc_dir, 'dec_' + value_class['sequence_name'][seq_idx] + '_' + \
                                            value_class['resolution'] + '_' + str(value_class['frameRate'][seq_idx]) + '_qp' + str(qp) + '.png')
                    os.system('bpgenc -f 444 -b 8 -q ' + str(qp) + ' ' + ori_file + ' -o ' + bin_file)
                    os.system('bpgdec -o ' + rec_file + ' ' + bin_file)
                    img_dec = Image.open(rec_file)
                    RGB_rec = np.array(img_dec)

                    Bits = os.path.getsize(bin_file) * 8
                    bits_list.append(Bits)

                    rgb_psnr, rgb_msssim = evaluate(RGB_ori, RGB_rec)
                    RGB_PSNR_list.append(rgb_psnr)
                    RGB_MSSSIM_list.append(rgb_msssim)
                    print('I frame, total_bits:[%d],PSNR_RGB:[%.4f],MSSSIM_RGB:[%.5f]' % (bits_list[0], RGB_PSNR_list[0], RGB_MSSSIM_list[0]))

                    images_prev_rec_tmp = RGB_rec[np.newaxis, :, :, :] / 255.0
                    images_prev_rec = np.zeros((1, self.image_size[0], self.image_size[1], 3, 4), np.float32)
                    for fr in range(4):
                        images_prev_rec[:, :, :, :, fr] = images_prev_rec_tmp[:, :, :, :]

                    flows_pre_rec = np.zeros((1, self.image_size[0], self.image_size[1], 2, 3), np.float32)
                    start_time = time.time()
                    for cur_indx in range(1, frames_to_be_encoded):
                        cur_ori_Y = ori_all_Y_list[cur_indx][np.newaxis, :, :, np.newaxis]
                        cur_ori_U = ori_all_U_list[cur_indx][np.newaxis, :, :, np.newaxis]
                        cur_ori_V = ori_all_V_list[cur_indx][np.newaxis, :, :, np.newaxis]
                        images_cur_ori = YUV2RGB420_custom(cur_ori_Y, cur_ori_U,cur_ori_V) / 255.0
                        images_pre_rec = images_prev_rec[:, :, :, :, 3]
                        if not ((self.image_size[0] > self.patch_height) and (self.image_size[1] > self.patch_width)):
                            flow_ori_test = self.sess.run(self.flow_ori_patch,
                                                          feed_dict={self.images_cur_ori_patch: images_cur_ori,
                                                                     self.images_pre_rec_patch: images_pre_rec})
                        else:
                            images_cur_ori_patches_list = reshape2patches_tesnsor(images_cur_ori, self.patches_h_list,
                                                                                  self.patches_w_list)
                            images_pre_rec_patches_list = reshape2patches_tesnsor(images_pre_rec, self.patches_h_list,
                                                                                  self.patches_w_list)
                            flow_ori_patches_list = []
                            for idx, (images_cur_ori_patch, images_pre_rec_patch) in enumerate(
                                    zip(images_cur_ori_patches_list, images_pre_rec_patches_list)):
                                flow_ori_patch = self.sess.run(self.flow_ori_patch,
                                                               feed_dict={
                                                                   self.images_cur_ori_patch: images_cur_ori_patch,
                                                                   self.images_pre_rec_patch: images_pre_rec_patch})
                                flow_ori_patches_list.append(flow_ori_patch)

                            flow_ori_test = reshape2image_tesnsor(flow_ori_patches_list, self.real_patches_h_list,
                                                                  self.real_patches_w_list, self.patch_num_heigt,
                                                                  self.patch_num_width)

                        bit_string_mvd_test, flow_pred_test, flow_diff_test, flow_diff_rec_0_test, flow_diff_rec_test, flow_rec_0_test, flow_rec_test, losses_mvd_test = self.sess.run(
                            [self.bit_string_mvd,
                             self.flow_pred,
                             self.flow_diff, self.flow_diff_rec_0,
                             self.flow_diff_rec, self.flow_rec_0, self.flow_rec, self._losses_mvd],
                            feed_dict={self.flow_ori_input: flow_ori_test,
                                       self.flows_pre_rec: flows_pre_rec,
                                       self.images_pre_rec_multiframes: images_prev_rec})
                        images_cur_pred_test = self.sess.run(
                            self.images_cur_pred,
                            feed_dict={self.images_pre_rec_multiframes: images_prev_rec,
                                       self.flow_rec_input: flow_rec_test,
                                       self.flows_pre_rec: flows_pre_rec})
                        bit_string_resi_test, bit_string_resi_dev_test, flow_3_pre_rec_test, flow_2_pre_rec_test, flow_1_pre_rec_test, images_cur_resi_test, images_cur_resi_rec_test, images_cur_rec_test, losses_resi_test, PSNR_list_test, MSSSIM_list_test = self.sess.run(
                            [self.bit_string_resi, self.bit_string_resi_dev, self.flow_3_pre_rec, self.flow_2_pre_rec,
                             self.flow_1_pre_rec,
                             self.images_cur_resi,
                             self.images_cur_resi_rec, self.images_cur_rec, self._losses_resi,
                             self._PSNR_list, self._MSSSIM_list],
                            feed_dict={self.images_pre_rec_multiframes: images_prev_rec,
                                       self.flow_rec_input: flow_rec_test,
                                       self.images_cur_ori: images_cur_ori,
                                       self.images_cur_pred_input: images_cur_pred_test,
                                       self.flows_pre_rec: flows_pre_rec})
                        mvd_bpp_test = losses_mvd_test[0]

                        resi_bpp_test = losses_resi_test[1]
                        resi_dev_bpp_test = losses_resi_test[2]
                        mvd_bits_info = int(mvd_bpp_test * width * height + 0.5)
                        resi_bits_info = int(resi_bpp_test * width * height + 0.5)
                        resi_dev_bits_info = int(resi_dev_bpp_test * width * height + 0.5)
                        cur_bits = mvd_bits_info + resi_bits_info + resi_dev_bits_info
                        bits_list.append(cur_bits)
                        RGB_PSNR_list.append(PSNR_list_test[0])
                        RGB_MSSSIM_list.append(MSSSIM_list_test[0])
                        if True:
                            print(
                                "cur_idx[%2d],time:[%4.4f],mvd_bits_info:[%d],resi_bits_info:[%d],resi_dev_bits_info:[%d],total_bits:[%d],PSNR_RGB:[%.4f],MSSSIM_RGB:[%.5f]"
                                % (cur_indx, time.time() - start_time,
                                   mvd_bits_info, resi_bits_info, resi_dev_bits_info, cur_bits,
                                   PSNR_list_test[0],
                                   MSSSIM_list_test[0]))
                        if self.args.visualize:
                            All_flows_to_vis = []
                            All_flows_to_vis.append(np.squeeze(flow_3_pre_rec_test[0] * 20))
                            All_flows_to_vis.append(np.squeeze(flow_2_pre_rec_test[0] * 20))
                            All_flows_to_vis.append(np.squeeze(flow_1_pre_rec_test[0] * 20))
                            All_flows_to_vis.append(np.squeeze(flow_ori_test[0] * 20))
                            All_flows_to_vis.append(np.squeeze(flow_pred_test[0] * 20))
                            All_flows_to_vis.append(np.squeeze(flow_diff_test[0] * 20))
                            All_flows_to_vis.append(np.squeeze(flow_diff_rec_0_test[0] * 20))
                            All_flows_to_vis.append(np.squeeze(flow_diff_rec_test[0] * 20))
                            All_flows_to_vis.append(np.squeeze(flow_rec_0_test[0] * 20))
                            All_flows_to_vis.append(np.squeeze(flow_rec_test[0] * 20))

                            All_RGB_images_to_vis = []
                            All_Gray_images_to_vis = []

                            RGB_frames = np.clip(images_prev_rec[:, :, :, :, 0] * 255, 0, 255).astype(np.uint8)
                            All_RGB_images_to_vis.append(np.squeeze(RGB_frames[0]))
                            RGB_frames = np.clip(images_prev_rec[:, :, :, :, 1] * 255, 0, 255).astype(np.uint8)
                            All_RGB_images_to_vis.append(np.squeeze(RGB_frames[0]))
                            RGB_frames = np.clip(images_prev_rec[:, :, :, :, 2] * 255, 0, 255).astype(np.uint8)
                            All_RGB_images_to_vis.append(np.squeeze(RGB_frames[0]))
                            RGB_frames = np.clip(images_prev_rec[:, :, :, :, 3] * 255, 0, 255).astype(np.uint8)
                            All_RGB_images_to_vis.append(np.squeeze(RGB_frames[0]))

                            RGB_frames = np.clip(images_cur_ori * 255, 0, 255).astype(np.uint8)
                            cur_ori_Gray_frames = cv2.cvtColor(np.squeeze(RGB_frames[0]), cv2.COLOR_RGB2GRAY)
                            All_RGB_images_to_vis.append(np.squeeze(RGB_frames[0]))

                            RGB_frames = np.clip(images_cur_pred_test * 255, 0, 255).astype(np.uint8)
                            cur_pred_Gray_frames = cv2.cvtColor(np.squeeze(RGB_frames[0]), cv2.COLOR_RGB2GRAY)
                            All_RGB_images_to_vis.append(np.squeeze(RGB_frames[0]))

                            RGB_frames = np.clip(images_cur_rec_test * 255, 0, 255).astype(np.uint8)
                            cur_rec_Gray_frames = cv2.cvtColor(np.squeeze(RGB_frames[0]), cv2.COLOR_RGB2GRAY)
                            All_RGB_images_to_vis.append(np.squeeze(RGB_frames[0]))
                            All_Gray_images_to_vis.append(
                                np.squeeze(np.clip((cur_ori_Gray_frames.astype(np.int64) - cur_pred_Gray_frames.astype(
                                    np.int64)) * 5.0 + 128 + 0.5,
                                                   0, 255).astype(np.uint8)))
                            All_Gray_images_to_vis.append(
                                np.squeeze(np.clip((cur_rec_Gray_frames.astype(np.int64) - cur_pred_Gray_frames.astype(
                                    np.int64)) * 5.0 + 128 + 0.5,
                                                   0, 255).astype(np.uint8)))

                            vis_flow_image_final([], All_flows_to_vis, All_RGB_images_to_vis, All_Gray_images_to_vis,
                                                 filename=os.path.join(self.args.exp_data_dir, 'figure',
                                                                       key_class + '_' +
                                                                       value_class['sequence_name'][
                                                                           seq_idx] + '_curidx' + str(
                                                                           cur_indx) + '.png'))

                        images_prev_rec[:, :, :, :, 0:3] = images_prev_rec[:, :, :, :, 1:4]
                        images_prev_rec[:, :, :, :, 3] = np.clip(images_cur_rec_test * 255 + 0.5, 0, 255).astype(
                            np.uint8) / 255.0
                        flows_pre_rec[:, :, :, :, 0:2] = flows_pre_rec[:, :, :, :, 1:3]
                        flows_pre_rec[:, :, :, :, 2] = flow_rec_test

                    Bpp_avg = np.mean(bits_list) / float(width * height)
                    RGB_PSNR_mean = np.mean(RGB_PSNR_list)
                    MSSSIM_avg = np.mean(RGB_MSSSIM_list)
                    print("Summary: Bpp: [%.4f],PSNR_RGB: [%.4f],MSSSIM_RGB: [%.5f]"
                          % (Bpp_avg, RGB_PSNR_mean, MSSSIM_avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tsd', '--test_seq_dir', type=str, default='/testSequence',
                        help='Directory containing test sequences')
    parser.add_argument('--test_class', type=str, default='ClassC',
                        help='Directory containing test sequences')
    parser.add_argument('-edd', '--exp_data_dir', type=str, required=True,
                        help='Directory containing experiment data')

    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of filters per layer.")

    parser.add_argument('--num_levels', type=int, default=4,
                        help='# of levels for feature extraction [6]')
    parser.add_argument('--warp_type', default='bilinear', choices=['bilinear', 'nearest'],
                        help='Warping protocol, [bilinear] or nearest')
    parser.add_argument('--use-dc', dest='use_dc', action='store_true',
                        help='Enable dense connection in optical flow estimator, [diabled] as default')
    parser.add_argument('--no-dc', dest='use_dc', action='store_false',
                        help='Disable dense connection in optical flow estimator, [disabled] as default')
    parser.set_defaults(use_dc=False)
    parser.add_argument('--output_level', type=int, default=3,
                        help='Final output level for estimated flow [4]')

    parser.add_argument('-v', '--visualize', dest='visualize', action='store_true',
                        help='Enable estimated flow visualization, [enabled] as default')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                        help='Disable estimated flow visualization, [enabled] as default')
    parser.set_defaults(visualize=True)
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='Learned parameter checkpoint file [None]')
    parser.add_argument('-rME', '--resumeMEnet', type=str, default=None,
                        help='Learned parameter checkpoint file [None]')
    parser.add_argument('-rMC', '--resumeMCnet', type=str, default=None,
                        help='Learned parameter checkpoint file [None]')

    parser.add_argument(
        "--command", choices=["train", "compress", "decompress"],
        help="What to do: 'train' loads training data and trains (or continues "
             "to train) a new model. 'compress' reads an image file (lossless "
             "PNG format) and writes a compressed binary file. 'decompress' "
             "reads a binary file and reconstructs the image (in PNG format). "
             "input and output filenames need to be provided for the latter "
             "two options.")
    parser.add_argument(
        "--mvd_N_filters", type=int, default=128,
        help="Number of filters per layer.")
    parser.add_argument(
        "--mvd_M_filters", type=int, default=192,
        help="Number of filters per layer.")
    parser.add_argument(
        "--resi_N_filters", type=int, default=128,
        help="Number of filters per layer.")
    parser.add_argument(
        "--resi_M_filters", type=int, default=192,
        help="Number of filters per layer.")
    parser.add_argument(
        "--lambda", type=int, default=16, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")

    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')

    tester = Tester(args)
    tester.test()