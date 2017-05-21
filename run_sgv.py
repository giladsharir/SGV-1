"""
Gilad Sharir (gilad@visualead.com) 2017

Adapted from tf-faster-rcnn and osvos-tensorflow repositories

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import scipy.misc
slim = tf.contrib.slim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(os.path.abspath(root_folder),"osvos-tf"))
import osvos
from dataset import Dataset

#Import FasterRCNN files
import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1



CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, ax, image_name, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def vis_masks(img, mask, ax):

    overlay_color = [255, 0, 0]
    transparency = 0.6


    mask = mask / np.max(mask)
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (
        overlay_color[0] * transparency + (1 - transparency) * img[:, :, 0])
    im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (
        overlay_color[1] * transparency + (1 - transparency) * img[:, :, 1])
    im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (
        overlay_color[2] * transparency + (1 - transparency) * img[:, :, 2])
    ax.imshow(im_over.astype(np.uint8))
    plt.axis('off')
    # plt.show()
    # plt.pause(0.01)


def vis_detections_masks(im, mask, ax, image_name, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))

    # mask
    overlay_color = [255, 0, 0]
    transparency = 0.6

    mask = mask / np.max(mask)
    im_over = np.ndarray(im.shape)
    im_over[:, :, 0] = (1 - mask) * im[:, :, 0] + mask * (
        overlay_color[0] * transparency + (1 - transparency) * im[:, :, 0])
    im_over[:, :, 1] = (1 - mask) * im[:, :, 1] + mask * (
        overlay_color[1] * transparency + (1 - transparency) * im[:, :, 1])
    im_over[:, :, 2] = (1 - mask) * im[:, :, 2] + mask * (
        overlay_color[2] * transparency + (1 - transparency) * im[:, :, 2])
    ax.imshow(im_over.astype(np.uint8))

    #detections
    # ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')


    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def faster_rcnn(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    fig, ax = plt.subplots(figsize=(12, 12))

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, ax, image_name, cls, dets, thresh=CONF_THRESH)






def sgv_test(sess, dataset, demonet, checkpoint_file, tfmodel, result_path, config=None):
    """Test one sequence
    Args:
    dataset: Reference to a Dataset object instance
    checkpoint_path: Path of the checkpoint to use for the evaluation
    result_path: Path to save the output images
    config: Reference to a Configuration object used in the creation of a Session
    Returns:
    """
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)

    # Input data
    batch_size = 1
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])

    # Create the cnn
    with slim.arg_scope(osvos.osvos_arg_scope()):
        net, end_points = osvos.osvos(input_image)
    probabilities = tf.nn.sigmoid(net)
    # global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

    # with g.as_default():
    #     with tf.device('/gpu:' + str(gpu_id)):
    #         with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(osvos.interp_surgery(tf.global_variables()))
    saver.restore(sess, checkpoint_file)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    #run osvos on all the frames

    for frame in range(0, dataset.get_test_size()):
        img, curr_img = dataset.next_batch(batch_size, 'test')
        curr_frame = curr_img[0].split('/')[-1].split('.')[0] + '.png'

        #test - osvos
        image = osvos.preprocess_img(img[0])
        res = sess.run(probabilities, feed_dict={input_image: image})
        res_np = res.astype(np.float32)[0, :, :, 0] > 162.0/255.0
        scipy.misc.imsave(os.path.join(result_path, curr_frame), res_np.astype(np.float32))
        # mask = res_np

        # fig, ax = plt.subplots(figsize=(12, 12))

        # vis_masks(img[0], mask, ax )
        # plt.imsave(os.path.join("output","mask_"+curr_frame),mask)


    #run faster-rcnn on all the frames
    if demonet == 'vgg16':
        net_rcnn = vgg16(batch_size=1)
    elif demonet == 'res101':
        net_rcnn = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net_rcnn.create_architecture(sess, "TEST", 21,
                                 tag='default', anchor_scales=[8, 16, 32])

    vlist = [v for v in tf.global_variables() if 'osvos' not in v.name.split('/')[0] and 'global_step' not in v.name]
    saver = tf.train.Saver(vlist)
    saver.restore(sess, tfmodel)
    # saver = tf.train.Saver()
    # saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    dataset.reset_iter()
    for frame in range(0, dataset.get_test_size()):
        img, curr_img = dataset.next_batch(batch_size, 'test')
        curr_frame = curr_img[0].split('/')[-1].split('.')[0] + '.png'

        #load mask

        mask = scipy.misc.imread(os.path.join(result_path, curr_frame))
        # mask = plt.imread(os.path.join("output", "mask_"+curr_frame))

        #convert image rgb --> bgr

        image = img[0][...,(2,1,0)]
        #test - faster rcnn
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(sess, net_rcnn, image)
        timer.toc()
        print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

        CONF_THRESH = 0.8
        NMS_THRESH = 0.3


        #save the mask + detections overlay
        fig, ax = plt.subplots(figsize=(12, 12))
        # vis_masks(img[0], mask, ax)

        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            vis_detections_masks(image, mask, ax, curr_frame, cls, dets, thresh=CONF_THRESH)



            # image = osvos.preprocess_img(img[0])
            # res = sess.run(probabilities, feed_dict={input_image: image})
            # res_np = res.astype(np.float32)[0, :, :, 0] > 162.0/255.0
            # scipy.misc.imsave(os.path.join(result_path, curr_frame), res_np.astype(np.float32))


        outputpath = 'output'
        plt.savefig(os.path.join(outputpath,curr_frame ))

        print('Saving ' + os.path.join(result_path, curr_frame))
# def sgv_test(net,image_name):






def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    gpu_id = 0

    #faster rcnn
    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('tf-faster-rcnn', 'output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    # init session
    sess = tf.Session(config=tfconfig)
    g = tf.Graph()
    # with g.as_default():
    #     with tf.device('/gpu:' + str(gpu_id)):
            # with tf.Session(config=tfconfig) as sess:




    process_osvos = True
    if process_osvos:
        #OSVOS test
        seq_name = "car-shadow"
        train_model = True
        result_path = os.path.join('osvos-tf', 'DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)


        # Train parameters
        parent_path = os.path.join('osvos-tf', 'models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
        logs_path = os.path.join('osvos-tf', 'models', seq_name)
        max_training_iters = 100
        # max_training_iters = 500

        # Define Dataset
        test_frames = sorted(os.listdir(os.path.join('osvos-tf', 'DAVIS', 'JPEGImages', '480p', seq_name)))
        test_imgs = [os.path.join('osvos-tf', 'DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in test_frames]
        if train_model:
            train_imgs = [os.path.join('osvos-tf', 'DAVIS', 'JPEGImages', '480p', seq_name, '00000.jpg')+' '+
                          os.path.join('osvos-tf', 'DAVIS', 'Annotations', '480p', seq_name, '00000.png')]
            dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
        else:
            dataset = Dataset(None, test_imgs, './')

        # Train the network
        if train_model:
            # More training parameters
            learning_rate = 1e-8
            save_step = max_training_iters
            side_supervision = 3
            display_step = 10
            # with g.as_default():
            #     with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                 save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)


        # tf.reset_default_graph()


        # with g.as_default():
        #     with tf.device('/gpu:' + str(gpu_id)):
        #         with tf.Session(config=tfconfig) as sess:
        checkpoint_path = os.path.join('osvos-tf', 'models', seq_name, seq_name+'.ckpt-'+str(max_training_iters))
        sgv_test(sess, dataset, demonet, checkpoint_path, tfmodel, result_path)
                # osvos.test(dataset, checkpoint_path, result_path)
    else:
        im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                    '001763.jpg', '004545.jpg']
        # with g.as_default():
        #     with tf.device('/gpu:' + str(gpu_id)):
                # with tf.Session(config=tfconfig) as sess:
        # sess.run(tf.global_variables_initializer())
        for im_name in im_names:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Demo for data/demo/{}'.format(im_name))
            # faster_rcnn(sess, net_rcnn, im_name)
            plt.savefig(im_name.split('.')[0] + ".png")


            # Show results
    # overlay_color = [255, 0, 0]
    # transparency = 0.6
    # plt.ion()
    # outputpath = "output"
    # for img_p in test_frames:
    #     frame_num = img_p.split('.')[0]
    #     img = np.array(Image.open(os.path.join('osvos-tf', 'DAVIS', 'JPEGImages', '480p', seq_name, img_p)))
    #     mask = np.array(Image.open(os.path.join(result_path, frame_num+'.png')))
    #     mask = mask/np.max(mask)
    #     im_over = np.ndarray(img.shape)
    #     im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
    #     im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1]*transparency + (1-transparency)*img[:, :, 1])
    #     im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])
    #     plt.imshow(im_over.astype(np.uint8))
    #     plt.axis('off')
    #     # plt.show()
    #     plt.pause(0.01)
    #     plt.savefig(os.path.join(outputpath,frame_num + ".png"))
    #     plt.clf()