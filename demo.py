# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1" #str(0)
import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
import tflearn
import numpy as np
import pprint
import pickle
import shutil

from modules.models_mvp2m import MeshNetMVP2M as MVP2MNet
from modules.models_p2mpp import MeshNet as P2MPPNet
from modules.config import execute
# from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict, load_demo_image
# from utils.visualize import plot_scatter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

folder = "test/plane"
n_images = 5

def main(cfg):
    # ---------------------------------------------------------------
    # Set random seed
    print('=> pre-porcessing')
    seed = 123
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    # ---------------------------------------------------------------
    num_blocks = 3
    num_supports = 2
    placeholders = {
        'features': tf.compat.v1.placeholder(tf.float32, shape=(None, 3), name='features'),
        'img_inp': tf.compat.v1.placeholder(tf.float32, shape=(n_images, 224, 224, 3), name='img_inp'),
        'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, 6), name='labels'),
        'support1': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support2': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support3': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'faces': [tf.compat.v1.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],
        'edges': [tf.compat.v1.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)],
        'lape_idx': [tf.compat.v1.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)],  # for laplace term
        'pool_idx': [tf.compat.v1.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks - 1)],  # for unpooling
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.compat.v1.placeholder(tf.int32),
        'sample_coord': tf.compat.v1.placeholder(tf.float32, shape=(43, 3), name='sample_coord'),
        'cameras': tf.compat.v1.placeholder(tf.float32, shape=(n_images, 5), name='Cameras'),
        'faces_triangle': [tf.compat.v1.placeholder(tf.int32, shape=(None, 3)) for _ in range(num_blocks)],
        'sample_adj': [tf.compat.v1.placeholder(tf.float32, shape=(43, 43)) for _ in range(num_supports)],
    }

    # step = cfg.test_epoch
    # root_dir = os.path.join(cfg.save_path, cfg.name)
    model1_dir = os.path.join('results', 'coarse_mvp2m', 'models')
    model2_dir = os.path.join('results', 'refine_p2mpp', 'models')
    # predict_dir = os.path.join(cfg.save_path, cfg.name, 'predict', str(step))
    # if not os.path.exists(predict_dir):
    #     os.makedirs(predict_dir)
    #     print('==> make predict_dir {}'.format(predict_dir))
    # -------------------------------------------------------------------
    print('=> build model')
    # Define model
    model1 = MVP2MNet(placeholders, logging=True, args=cfg)
    model2 = P2MPPNet(placeholders, logging=True, args=cfg)
    # ---------------------------------------------------------------
    print('=> load data')
    demo_img_list = ['data/{}/image_0.png'.format(folder),
                     'data/{}/image_1.png'.format(folder),
                     'data/{}/image_2.png'.format(folder),
                     'data/{}/image_3.png'.format(folder),
                     'data/{}/image_4.png'.format(folder)]
    img_all_view = load_demo_image(demo_img_list, n_images)
    cameras = np.loadtxt('data/{}/cameras.txt'.format(folder))
    # data = DataFetcher(file_list=cfg.test_file_path, data_root=cfg.test_data_path, image_root=cfg.test_image_path, is_val=True)
    # data.setDaemon(True)
    # data.start()
    # ---------------------------------------------------------------
    print('=> initialize session')
    sesscfg = tf.compat.v1.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.compat.v1.Session(config=sesscfg)
    sess.run(tf.compat.v1.global_variables_initializer())
    # sess2 = tf.Session(config=sesscfg)
    # sess2.run(tf.global_variables_initializer())
    # ---------------------------------------------------------------
    model1.load(sess=sess, ckpt_path=model1_dir, step=50)
    model2.load(sess=sess, ckpt_path=model2_dir, step=10)
    # exit(0)
    # ---------------------------------------------------------------
    # Load init ellipsoid and info about vertices and edges
    pkl = pickle.load(open('data/iccv_p2mpp.dat', 'rb'))
    # Construct Feed dict
    feed_dict = construct_feed_dict(pkl, placeholders)
    # ---------------------------------------------------------------
    tflearn.is_training(False, sess)
    print('=> start test stage 1')
    feed_dict.update({placeholders['img_inp']: img_all_view})
    feed_dict.update({placeholders['labels']: np.zeros([10, 6])})
    feed_dict.update({placeholders['cameras']: cameras})
    stage1_out3 = sess.run(model1.output3, feed_dict=feed_dict)
    
    print('=> start test stage 2')
    feed_dict.update({placeholders['features']: stage1_out3})
    vert = sess.run(model2.output2l, feed_dict=feed_dict)
    vert = np.hstack((np.full([vert.shape[0],1], 'v'), vert))
    face = np.loadtxt('data/face3.obj', dtype='|S32')
    mesh = np.vstack((vert, face))

    pred_path = 'data/{}/predict.obj'.format(folder)
    np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')

    print('=> save to {}'.format(pred_path))

if __name__ == '__main__':
    print('=> set config')
    args=execute()
    # pprint.pprint(vars(args))
    main(args)
