# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import numpy as np
import pickle
import scipy.sparse as sp
import networkx as nx
import threading
import queue
import sys
import cv2
import math
import time
import os
import glob
from PIL import Image

from azureml.core import Dataset
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azureml.fsspec import AzureMachineLearningFileSystem


np.random.seed(123)


class DataFetcher(threading.Thread):
    def __init__(self, file_list, data_root, image_root, is_val=False, mesh_root=None):
        super(DataFetcher, self).__init__()
        
        ml_client = MLClient.from_config(credential=DefaultAzureCredential())
        p2mppdata = ml_client.data.get("p2mppdata", version="2")
        shapenet_rendering = ml_client.data.get("ShapeNetRendering", version="1")

        self.fs_p2mppdata = AzureMachineLearningFileSystem(p2mppdata.path)
        self.fs_shapenet_rendering = AzureMachineLearningFileSystem(shapenet_rendering.path)

        print(self.fs_p2mppdata.ls()[0])
        print(self.fs_shapenet_rendering.ls()[0])

        self.stopped = False
        self.queue = queue.Queue(64)
        self.data_root = data_root
        self.image_root = image_root
        self.is_val = is_val

        self.pkl_list = []
        with open(file_list, 'r') as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                self.pkl_list.append(line)
        self.index = 0
        self.mesh_root = mesh_root
        self.number = len(self.pkl_list)
        np.random.shuffle(self.pkl_list)

    def work(self, idx):
        pkl_item = self.pkl_list[idx]
        pkl_path = os.path.join(self.data_root, pkl_item)

        with self.fs_p2mppdata.open(pkl_path, "rb") as f:
            pkl = pickle.load(f, encoding='bytes')
        if self.is_val:
            label = pkl[1]
        else:
            label = pkl
        # load image file
        img_root = self.image_root
        ids = pkl_item.split('_')
        category = ids[-3]
        item_id = ids[-2]
        img_path = os.path.join(img_root, category, item_id, 'rendering')
        metadata_path = os.path.join(img_path, "rendering_metadata.txt")
        with self.fs_shapenet_rendering.open(metadata_path, "r") as f:
            camera_meta_data = np.loadtxt(f)

        if self.mesh_root is not None:
            mesh_path = os.path.join(self.mesh_root, category + '_' + item_id + '_00_predict.xyz')
            with self.fs_shapenet_rendering.open(mesh_path, "r") as f:
                mesh = np.loadtxt(f)
        else:
            mesh = None
        imgs = np.zeros((3, 224, 224, 3))
        poses = np.zeros((3, 5))
        for idx, view in enumerate([0, 6, 7]):
            img_file_path = os.path.join(img_path, str(view).zfill(2) + '.png')
            with self.fs_shapenet_rendering.open(img_file_path, "rb") as f:
                img = Image.open(f)
                img = np.array(img)
            img[np.where(img[:, :, 3] == 0)] = 255
            img = cv2.resize(img, (224, 224))
            img_inp = img.astype('float32') / 255.0
            imgs[idx] = img_inp[:, :, :3]
            poses[idx] = camera_meta_data[view]
        return imgs, label, poses, pkl_item, mesh

    def run(self):
        while self.index < 9000000 and not self.stopped:
            self.queue.put(self.work(self.index % self.number))
            self.index += 1
            if self.index % self.number == 0:
                np.random.shuffle(self.pkl_list)

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()


if __name__ == '__main__':
    file_list = sys.argv[1]
    data = DataFetcher(file_list)
    data.start()
    data.stopped = True
