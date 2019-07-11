# define the class to load the caltech101
# version 2 update:
#   1. use skimage + plt solution, including imread, resize, plt.imshow
#   2. use zero padding
#   3. compute the mean, zero center
#
#
# version 3 update:
#   1. make a new load_set
import tensorflow as tf


# import matplotlib
# matplotlib.use("Agg")

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image, ImageOps

from keras.preprocessing import image

##### For one-hot label
from keras.utils import np_utils

class SUN397:

    def __init__(self, data_dir):
        self.nb_classes = 397
        self.img_depth = 3
        self.data_dir = data_dir + 'SUN397'
        self.train_img_file = '/Patitions/Training_01.txt'
        self.test_img_file = '/Patitions/Testing_01.txt'
        self.classes_name_list = '/ClassName.txt'
        self.train_label_file ='/ClassName.txt'
        self.test_label_file = '/ClassName.txt'

        self.produce_label_map()

    def produce_label_map(self):
        # begin to tansfer the str label to int label
        print('Begin to create a map to tansfer the str label to int label...')
        class_name_file = self.data_dir + self.classes_name_list
        class_str = [str(line.strip()) for line in open(class_name_file).readlines()]
        class_count = len(class_str)
        print('%d class names are loaded' % class_count)
        # begin to create the map
        self.le = preprocessing.LabelEncoder()
        self.le.fit(class_str)
        print(list(self.le.classes_))
        print('Label map created...')

    def compute_meam(self, x_train, x_test):
        x = np.vstack((x_train, x_test))

        mean = [None] * self.img_depth
        for i in range(self.img_depth):
            mean[i] = np.mean(x[:, :, :, i])
            #print(mean)
        return mean

    def zero_center(self, mean, x_train, x_test):
        for idx_sp in range(len(x_train)):
            for idx_cha in range(self.img_depth):
                x_train[idx_sp][:,:,idx_cha] = x_train[idx_sp][:,:,idx_cha] - mean[idx_cha]

        for idx_sp in range(len(x_test)):
            for idx_cha in range(self.img_depth):
                x_test[idx_sp][:,:,idx_cha] = x_test[idx_sp][:,:,idx_cha] - mean[idx_cha]

        return x_train, x_test

    def load_img_file(self, file, desired_img_dim):
        print('Loading image file %s' % file)
        start_time_ = time.time()
        img_file_path = [str(line.strip()) for line in open(file).readlines()]  # read all the image file name
        nb_sample = len(img_file_path)
        print('Image count: %d' % nb_sample)

        data_resized_holder = np.empty([nb_sample, desired_img_dim, desired_img_dim, self.img_depth], dtype='float32')

        for idx in range(nb_sample):
            img_file1 = self.data_dir + img_file_path[idx].replace("\\", "/")     # the image file path
            # print(str(img_file1))
            # 1. read the image
            img1 = image.load_img(img_file1)

            # 2. resize
            img1 = img1.resize((desired_img_dim, desired_img_dim), resample=0)


            # 6. give to the holder
            data_resized_holder[idx] = img1
            if(idx % 1000==0):
                print('%d image loaded.' % idx)

        print('\nImage file loaded, the shape is ' + str(data_resized_holder.shape))

        return data_resized_holder, nb_sample


    def load_label_file(self, file, one_hot=True):
        # loading the training labels
        print('Loading label file %s' % file)
        label_str = [str(line.strip()) for line in open(file).readlines()]
        nb_unique = len(label_str)
        labels_unique = self.le.transform(label_str)
        # print(labels_unique)
        labels_holder = np.hstack((  [ labels_unique[i] ] * 50 for i in range(nb_unique)))
        # print(labels_holder)
        nb_sample = len(labels_holder)
        if one_hot == True:
            labels = np.array([[float(i == l) for i in range(self.nb_classes)] for l in labels_holder])
        else:
            labels = labels_holder
        print('Labels loaded, shape is:' + str(labels.shape))


        return labels, nb_sample


    def load(self, desired_image_dim, one_hot= True):

        ############################################ loading training data##################################################
        print('\nBegin to load training data...\n')
        train_img_file_path = self.data_dir + self.train_img_file
        train_label_file_path = self.data_dir + self.train_label_file

        x_train, nb_train_sample_1 = self.load_img_file(train_img_file_path, desired_image_dim)
        y_train, nb_train_sample_2 = self.load_label_file(train_label_file_path, one_hot=one_hot)
        assert nb_train_sample_1 == nb_train_sample_2, "The train img and label samples are not matched..."
        self.nb_train_sample = nb_train_sample_1

        print('\nTraining data loaded.\n')

        ############################################ loading testing data##################################################
        print('\nBegin to load testing data...\n')
        test_img_file_path = self.data_dir + self.test_img_file
        test_label_file_path = self.data_dir + self.test_label_file

        x_test, nb_test_sample_1 = self.load_img_file(test_img_file_path, desired_image_dim)
        y_test, nb_test_sample_2 = self.load_label_file(test_label_file_path, one_hot=one_hot)
        assert nb_test_sample_1 == nb_test_sample_2, "The test img and label samples are not matched..."
        self.nb_test_sample = nb_test_sample_1

        print('\nTesting data loaded.\n')

        return x_train, y_train, self.nb_train_sample, x_test, y_test, self.nb_test_sample


    def load_set(self, desired_image_dim, nb_tr_per_class=50, one_hot=True):

        x_train, y_train, self.nb_tr_sample, x_test, y_test, self.nb_te_sample = self.load(desired_image_dim, one_hot=one_hot)

        return x_train, y_train, self.nb_train_sample, x_test, y_test, self.nb_test_sample

if __name__ == '__main__':
    DATA_DIR = '/media/jeremyfeng/volumeD/dataSet/'
    sun397 = SUN397(DATA_DIR)
    sun397.load_set(desired_image_dim=224, one_hot=False)

