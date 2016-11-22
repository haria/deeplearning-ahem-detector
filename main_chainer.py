"""
MIT License
Copyright (c) 2016 Francesco Gadaleta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

--------------------------------------------------------------------------------------
Note:
Please build training/testing set before running this script.
Make sure to create the local path that have been hardcoded in the following scripts,
then execute

    % python make_data_class_0.py
    % python make_data_class_1.py
---------------------------------------------------------------------------------------

"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.style as ms
ms.use('seaborn-muted')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
"""
###########################################################
import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
###########################################################

from os import listdir
from os.path import isfile, join
import utils as ut

import librosa
import librosa.display
import IPython.display
import numpy as np
from skimage.measure import block_reduce
import skimage.io as io

from AhemNet import AhemNet


def load_image(filename):
    img = io.imread(filename)
    img = img.transpose((2, 0, 1))
    img = img[:3, :, :]
    return img


# network configuration
batch_size = 32
# number of epochs
nb_epoch = 5
# number of convolutional filters to use
nb_filters = 32
# number of classes
nb_classes = 2
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
# we save generated images here (make sure there is space)
path_class_0 = 'data/class_0/'
path_class_1 = 'data/class_1/'

# load filenames into lists
class0_files = [f for f in listdir(path_class_0) if isfile(join(path_class_0, f))]
class1_files = [f for f in listdir(path_class_1) if isfile(join(path_class_1, f))]

# prepare training set
X_t = []
Y_t = []

#for fn in class0_files[:100]:
for fn in class0_files:
    imagepath = os.path.join(path_class_0, fn)
    img = load_image(imagepath)
    X_t.append(img)
    Y_t.append(0)

#for fn in class1_files[:100]:
for fn in class1_files:
    imagepath = os.path.join(path_class_1, fn)
    img = load_image(imagepath)
    X_t.append(img)
    Y_t.append(1)

X_t = np.asarray(X_t)
X_t = X_t.astype('float32')
X_t /= 255

Y_t = np.asarray(Y_t)
Y_t = Y_t.astype('int32')
#Y_t = np_utils.to_categorical(Y_t, nb_classes)

indexes=np.asarray(range(0, len(X_t)))
np.random.shuffle(indexes)

X_t=X_t[indexes]
Y_t=Y_t[indexes]

img_rows, img_cols = X_t.shape[2], X_t.shape[3]
# input image dimensions
img_channels = 3               # RGB
input_shape = (3, img_rows, img_cols)

"""
## test set
X_test = []
Y_test = []

for fn in class0_files[6000:8000]:
    img = io.imread(os.path.join(path_class_0, fn))
    img = img.transpose((2,0,1))
    img = img[:3, :, :]
    X_test.append(img)
    Y_test.append(0)

for fn in class1_files[6000:8000]:
    img = io.imread(os.path.join(path_class_1, fn))
    img = img.transpose((2,0,1))
    img = img[:3, :, :]
    X_test.append(img)
    Y_test.append(1)

X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)
X_test = X_test.astype('float32')
X_test /= 255

Y_test = np_utils.to_categorical(Y_test, nb_classes)
"""

"""
def make_model():
    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model
"""


import argparse
import chainer.cuda as cuda
from Classifier import Classifier
from Train import Train
##########################################################
# parse argument input from user
parser = argparse.ArgumentParser(description='Chainer AhemNet')
parser.add_argument('--outprefix', default='AhemDetector', help='Prefix of path to save model and state after each epoch')
parser.add_argument('--gpu', default=0, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', default=20, type=int, help='number of epochs to learn')
parser.add_argument('--batchsize', type=int, default=4, help='learning minibatch size')
parser.add_argument('--numclasses', type=int, default=2, help='input size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
args = parser.parse_args()

##########################################################
# cupy or numpy
if cuda.available and args.gpu >= 0:
    print("CUDA is enabled, use device %d." % args.gpu)
    cuda.get_device(args.gpu).use()
    xp = cuda.cupy
else:
    print ("CUDA is disabled, use CPU.")
    xp = np

##########################################################
model = Classifier(AhemNet(num_classes=args.numclasses))

if cuda.available:
    model.to_gpu()

##########################################################
print("Optimizer Init.")
print("Learning rate: %f." % args.lr)
optimizer = chainer.optimizers.SGD(lr=args.lr)
#optimizer = chainer.optimizers.AdaDelta()
optimizer.setup(model)

##########################################################
# load data
print("Load Dataset.")

trainer = Train(model, optimizer, xp)
trainer.loop_train(X_t, Y_t, batch_size=args.batchsize, max_epoch=args.epoch, prefix=args.outprefix,
                   cur_epoch=0)
##########################################################
