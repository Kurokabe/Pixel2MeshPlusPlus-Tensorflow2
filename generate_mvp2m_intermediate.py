# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# import tflearn
import numpy as np
import pprint
import pickle
import shutil
import os

from modules.models_mvp2m import MeshNetMVP2M
from modules.config import execute
from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict
from utils.visualize import plot_scatter


def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    # ---------------------------------------------------------------
    # Set random seed
    print("=> pre-porcessing")
    seed = 123
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    # ---------------------------------------------------------------
    num_blocks = 3
    num_supports = 2
    placeholders = {
        "features": tf.compat.v1.placeholder(
            tf.float32, shape=(None, 3), name="features"
        ),
        "img_inp": tf.compat.v1.placeholder(
            tf.float32, shape=(3, 224, 224, 3), name="img_inp"
        ),
        "labels": tf.compat.v1.placeholder(tf.float32, shape=(None, 6), name="labels"),
        "support1": [
            tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)
        ],
        "support2": [
            tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)
        ],
        "support3": [
            tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)
        ],
        "faces": [
            tf.compat.v1.placeholder(tf.int32, shape=(None, 4))
            for _ in range(num_blocks)
        ],
        "edges": [
            tf.compat.v1.placeholder(tf.int32, shape=(None, 2))
            for _ in range(num_blocks)
        ],
        "lape_idx": [
            tf.compat.v1.placeholder(tf.int32, shape=(None, 10))
            for _ in range(num_blocks)
        ],  # for laplace term
        "pool_idx": [
            tf.compat.v1.placeholder(tf.int32, shape=(None, 2))
            for _ in range(num_blocks - 1)
        ],  # for unpooling
        "dropout": tf.compat.v1.placeholder_with_default(0.0, shape=()),
        "num_features_nonzero": tf.compat.v1.placeholder(tf.int32),
        "sample_coord": tf.compat.v1.placeholder(
            tf.float32, shape=(43, 3), name="sample_coord"
        ),
        "cameras": tf.compat.v1.placeholder(tf.float32, shape=(3, 5), name="Cameras"),
        "faces_triangle": [
            tf.compat.v1.placeholder(tf.int32, shape=(None, 3))
            for _ in range(num_blocks)
        ],
        "sample_adj": [
            tf.compat.v1.placeholder(tf.float32, shape=(43, 43))
            for _ in range(num_supports)
        ],
    }

    step = cfg.test_epoch
    root_dir = os.path.join(cfg.save_path, cfg.name)
    model_dir = os.path.join(cfg.save_path, cfg.name, "models")
    predict_dir = os.path.join(
        cfg.save_path, cfg.name, "coarse_intermediate", str(step)
    )
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
        print("==> make predict_dir {}".format(predict_dir))
    # -------------------------------------------------------------------
    print("=> build model")
    # Define model
    model = MeshNetMVP2M(placeholders, logging=True, args=cfg)
    # ---------------------------------------------------------------
    print("=> load data")
    data = DataFetcher(
        file_list=cfg.coarse_result_file_path,
        data_root=cfg.coarse_result_data_path,
        image_root=cfg.coarse_result_image_path,
        is_val=False,
    )
    data.setDaemon(True)
    data.start()
    # ---------------------------------------------------------------
    print("=> initialize session")
    sesscfg = tf.compat.v1.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.compat.v1.Session(config=sesscfg)
    sess.run(tf.compat.v1.global_variables_initializer())
    # ---------------------------------------------------------------
    model.load(sess=sess, ckpt_path=model_dir, step=step)
    # ---------------------------------------------------------------
    # Load init ellipsoid and info about vertices and edges
    pkl = pickle.load(open("data/iccv_p2mpp.dat", "rb"))
    # Construct Feed dict
    feed_dict = construct_feed_dict(pkl, placeholders)
    # ---------------------------------------------------------------
    test_number = data.number
    tf.keras.backend.set_learning_phase(1)
    print("=> start test stage 1")
    for iters in range(test_number):
        # Fetch training data
        # need [img, label, pose(camera meta data), dataID]
        img_all_view, labels, poses, data_id, mesh = data.fetch()
        feed_dict.update({placeholders["img_inp"]: img_all_view})
        feed_dict.update({placeholders["labels"]: labels})
        feed_dict.update({placeholders["cameras"]: poses})
        # ---------------------------------------------------------------
        out1, out2, out3 = sess.run(
            [model.output1, model.output2, model.output3], feed_dict=feed_dict
        )
        # ---------------------------------------------------------------
        # save GT
        label_path = os.path.join(predict_dir, data_id.replace(".dat", "_ground.xyz"))
        np.savetxt(label_path, labels)
        # save 1
        # out1_path = os.path.join(predict_dir, data_id.replace('.dat', '_predict_1.xyz'))
        # np.savetxt(out1_path, out1)
        # # save 2
        # out2_path = os.path.join(predict_dir, data_id.replace('.dat', '_predict_2.xyz'))
        # np.savetxt(out2_path, out2)
        # save 3
        out3_path = os.path.join(predict_dir, data_id.replace(".dat", "_predict.xyz"))
        np.savetxt(out3_path, out3)

        print("Iteration {}/{}, Data id {}".format(iters + 1, test_number, data_id))

    # ---------------------------------------------------------------
    data.shutdown()
    print("CNN-GCN Optimization Finished!")


if __name__ == "__main__":
    print("=> set config")
    args = execute()
    pprint.pprint(vars(args))
    main(args)
