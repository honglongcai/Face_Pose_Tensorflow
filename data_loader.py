import os

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

class DataLoader(object):
    def __init__(self, image_path):
        self.image_path = image_path
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.read_images()

    def __call__(self, mode, batch_size=1):
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.Y_train))
            dataset = dataset.shuffle(1000).repeat()
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.Y_test))
        dataset = dataset.batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    def read_images(self):
        X, Y = [], []
        dirs = os.listdir(self.image_path)
        for dir in dirs:
            images = os.listdir(os.path.join(self.image_path, dir))
            for image in images:
                X.append(cv2.imread(os.path.join(self.image_path, dir, image)))
                Y.append(int(dir))
        X, Y = np.asarray(X).astype(np.float), np.asarray(Y)
        X = X / np.max(X)
        num = X.shape[0]
        num_train = int(num * 0.9)
        idr = np.arange(num_train)
        np.random.shuffle(idr)
        X_train = X[idr[:num_train], :]
        X_test = X[idr[num_train:], :]
        Y_train = Y[idr[:num_train]]
        Y_test = Y[idr[num_train:]]
        return X_train, Y_train, X_test, Y_test



