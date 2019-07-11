from keras.applications.resnet50 import  preprocess_input
from resnet50 import ResNet50


import keras
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras.regularizers import l2

import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import tensorflow as tf

import os
import argparse
import datetime
import socket



from sun397 import SUN397

# Parse the commandline argument
parser = argparse.ArgumentParser(description='Recomputation of Dense Layers.')
parser.add_argument('-g', '--igpu', type=int, default=0,
                    help='Index of gpu')
parser.add_argument('-t','--itest', type=str, default='default',
                    help='Index of experiment.')
parser.add_argument('-r','--irc', type=int, default=1,
                    help='Setting of ReComp. 0: w/ o ReComp; '
                         '                   1: w/ ReComp.')
parser.add_argument('-i','--iinter', type=int, default=1,
                    help='The interval epochs that the ReComp method executes. 1 represents that the ReComp method executes on each epoch.')
parser.add_argument('-l','--iloop', type=int, default=3,
                    help='Number of loop that the ReComp method takes during each execution session.')
# parser.add_argument('-d','--idatadir', type='str', default='D\:\/dataSet\/',
#                     help='The root directory of you datasets, which contains the sun397 dataset folder.')
args = parser.parse_args()

# GPU SELECTION
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.igpu)

# TEST SETTING
TEST_SETTING = args.itest
if args.irc in (0,1):
    RC_SETTING = args.irc
else:
    print('Please input proper setting of ReComp. 0: w/ o ReComp; 1: w/ ReComp.')

RC_EPOCH = args.iinter
RC_LOOP_PER_EPOCH = args.iloop

# NET / DATASET
MODEL = 'resnet50'
DATASET = 'sun397'


# DATA_DIR
DATA_DIR = ''                ########## Please revise the directory.  ##########

MACHINE_NAME = socket.gethostname()

# Print the program settings
print('Program detail:\n'
      'Date: {}\n'
      'Machine: {}\n'
      'MODEL: {}\n'
      'DATASET: {}\n'
      'TEST_SETTING: {}\n'
      'RC_SETTING: {}\n'
      .format(datetime.datetime.now(), MACHINE_NAME, MODEL, DATASET, TEST_SETTING, RC_SETTING))


# Parameter setting
IMAGE_SIZE = 224
FC1_DIM = 2048
NB_CLASSES = 397


def rand_weight_drop(weight, shape, keep_rate):
    nb_weight_elem = shape[0] * shape[1]
    nb_kp = int(round(nb_weight_elem * keep_rate))
    nb_dp = nb_weight_elem - nb_kp
    kp_ = np.ones([nb_kp, ])
    dp_ = np.zeros([nb_dp, ])
    kp_dp = np.concatenate([kp_, dp_])
    np.random.shuffle(kp_dp)
    kp_dp_reshape = np.reshape(kp_dp, weight.shape)
    weight_drop = tf.multiply(weight, kp_dp_reshape)

    return weight_drop

def pinv_func2(H, omega=1.):
    identity = tf.constant(np.identity(H.shape[1]), dtype=tf.float32)
    H_T = tf.transpose(H)
    H_ = tf.matmul(tf.matrix_inverse(tf.matmul(H_T, H) + identity / omega), H_T)

    return H_

def build_resnet50(nb_classes,
                        weights,
                        weight_decay = 1e-4):
    print('Building the resnet50 model...')

    img_rows, img_cols = IMAGE_SIZE, IMAGE_SIZE
    img_channels = 3

    img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (
    img_rows, img_cols, img_channels)


    if weights == 'imagenet':
        base_model = ResNet50(input_shape=img_dim,
                                 weights='imagenet',
                                 include_top=True)
        x = Dense(nb_classes, kernel_regularizer= l2(weight_decay), activation='softmax', use_bias=False, name='fc')(base_model.get_layer('flatten').output)
        model = Model(inputs=base_model.input, outputs=x)

    elif weights is not None:
        base_model = ResNet50(input_shape=img_dim,
                                 weights=None,
                                 include_top=True)
        x = Dense(nb_classes, kernel_regularizer= l2(weight_decay), activation='softmax', use_bias=False, name='fc')(base_model.get_layer('flatten').output)
        model = Model(inputs=base_model.input, outputs=x)
        model.load_weights(weights, by_name=True)

    elif weights is None:
        base_model = ResNet50(input_shape=img_dim,
                                 weights=None,
                                 include_top=True)
        x = Dense(nb_classes, kernel_regularizer= l2(weight_decay), activation='softmax', use_bias=False, name='fc')(base_model.get_layer('flatten').output)
        model = Model(inputs=base_model.input, outputs=x)


    # model.summary()
    print("Model created")

    return model


def train_model():
    nb_epochs = 10
    learning_rate = 0.001
    bp_lr_list = [0.001, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001]
    # rl_ur_list = [0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
    rl_ur_list = [1, 1, 1, 1, 0.95, 0.95, 0.95, 0.95 , 0.9, 0.9]
    dl_batch_size = 64
    init_epoch = 1

    ########### prepare the data ###########
    print('Preparing data...')
    print('Loading image...')
    sun397 = SUN397(DATA_DIR)
    x_train, y_train, nb_tr_sample, x_test, y_test, nb_te_sample = sun397.load_set(desired_image_dim=IMAGE_SIZE,
                                                                                   one_hot=True)

    print('Preprocessing img...')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)

    print('Data loaded.')

    print('Building the model...\n')
    model = build_resnet50(NB_CLASSES,
                        weights='imagenet',
                        weight_decay=1e-4)
    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    model.summary()
    print('Model compiled...\n\n')

    with tf.device('/gpu:0'):
        with tf.name_scope('config_para'):
            omega = 1.
            weights_elem_keep_rate = 1.0
            rl_ur_ph = tf.placeholder(tf.float32, name='rl_ur_ph')

        with tf.name_scope('forward_pass'):
            y_ = tf.placeholder(tf.float32, [None, NB_CLASSES], name='y_ph')

            w_fc12_ph = tf.placeholder(tf.float32, [FC1_DIM, NB_CLASSES], name='w_fc12_ph')

            fc1_ph = tf.placeholder(tf.float32, [None, FC1_DIM], name='fc1_ph')
            fc2_ph = tf.placeholder(tf.float32, [None, NB_CLASSES], name='fc2_ph')

        with tf.name_scope('recomputation_learning'):
            y_e = tf.subtract(y_, fc2_ph, name='y_e')
            w_fc12_rl_ts = tf.matmul(pinv_func2(fc1_ph), y_e, name='w_fc12_rl_ts')

            w_fc12_rl_dp_ts = rand_weight_drop(w_fc12_rl_ts, shape=[FC1_DIM, NB_CLASSES],
                                               keep_rate=weights_elem_keep_rate)
            w_fc12_rl_ts = w_fc12_ph + w_fc12_rl_dp_ts * rl_ur_ph

    #########################
    #          loop         #
    #########################
    for epoch in range(init_epoch, nb_epochs + 1):
        #########################
        # learning rate control #
        #########################
        K.set_value(model.optimizer.lr, bp_lr_list[epoch - 1])

        #########################
        #     training step     #
        #########################
        print('Epoch %d: BP learning rate: %s; Reverse learning update rate: %s. ' % (
        epoch, K.get_value(model.optimizer.lr), rl_ur_list[epoch - 1]))
        hist = model.fit(x_train, y_train, batch_size=dl_batch_size,
                      epochs=epoch, verbose=2,
                      validation_data=(x_test, y_test),
                      shuffle=True, initial_epoch=epoch-1)


        ############################
        #     REVERSE LEARNING     #
        ############################
        if ( (epoch - 1) % RC_EPOCH == 0 ) & ( (epoch - 1) >= 0 ) & (RC_SETTING!=0):
            # 1. Extract the fc1 features
            print('\nEpoch {}: Extracting the training feature...'.format(epoch))
            model_fl = Model(model.input, outputs=model.get_layer(name='flatten').output)
            fc1_tr_np = model_fl.predict(x_train)
            fc2_tr_np = model.predict(x_train)


            for idx_RLloop in range(RC_LOOP_PER_EPOCH):
                # 2. Extract the fc12 weight
                print('Epoch {}, idx_RCloop {}: Extracting the fc12 weights...'.format(epoch, idx_RLloop))
                w_fc12_np = model.get_layer(name='fc').get_weights()[0]

                # 3. Update the fc12 weight
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    w_fc12_rl_np = w_fc12_rl_ts.eval(
                        feed_dict={
                            w_fc12_ph: w_fc12_np,
                            fc1_ph: fc1_tr_np,
                            fc2_ph: fc2_tr_np,
                            y_: y_train,
                            rl_ur_ph: rl_ur_list[epoch-1]
                        })

                # 4. Set back the fc12 weight
                w_fc12_rl_list = [None]
                w_fc12_rl_list[0] = w_fc12_rl_np
                model.get_layer(name='fc').set_weights(w_fc12_rl_list)

            eval_loss, eval_acc = model.evaluate(x_test,y_test, verbose=0, batch_size=64)
            print('Epoch {}: After ReComp, acc: {}\n'.format(epoch, eval_acc))

    return

if __name__ == '__main__':
    train_model()
