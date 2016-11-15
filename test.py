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

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json

import skimage.io as io

from os import listdir
from os.path import isfile, join
import utils as ut

import librosa
import librosa.display
import IPython.display
import numpy as np
from skimage.measure import block_reduce
import skimage.io as io

# network configuration
batch_size = 128

def load_image(filename):
    img = io.imread(filename)
    img = img.transpose((2, 0, 1))
    img = img[:3, :, :]
    return img

print ("load model")
from keras.models import load_model
model = load_model('ahemdetector.h5')

# the data, shuffled and split between train and test sets
#path_newsample = './data/new_sample/'
path_newsample = './data/ahem_sample/'
newsample_files = [join(path_newsample, f) for f in listdir(path_newsample) if isfile(join(path_newsample, f))]

# prepare test set as we did for training set
X_test = []

print ("load data.")
for fn in newsample_files:
    img = load_image(fn)
    X_test.append(img)

X_test = np.asarray(X_test)
X_test = X_test.astype('float32')
X_test /= 255
print ("predict data")
# grab a large cup of coffee this will take a while
predictions = model.predict_classes(X_test)

print ("get result")
# collect all indices of noisy samples (class 1)
# start position is encoded in filename (a trick to run this in parallel with no sequential order)
noisy_frames = np.where(predictions==1)[0]
noisy_files = [newsample_files[n] for n in noisy_frames]

print (noisy_frames)
print ("done")
exit(1)
# Load a sound with a lot of "ahem" in it
path = './data'
sound_file_paths = [os.path.join(path, "provocation_dirty.wav")]
sound_names = ["dirty"]
raw_sounds = ut.load_sound_files(sound_file_paths)
windowsize = 6000
# create positive samples
audiosamples = raw_sounds[0]
numsamples = audiosamples.shape[0]

original_audio = audiosamples
clean_audio = audiosamples

# Playback from ipython (cool uh?)
IPython.display.Audio(data=original_audio, rate=44100)

noisy_start = []
for fn in noisy_files:
    noisy_start.append(int(fn.split('_')[2].split('.')[0]))

noisy_start.sort(reverse=True)

clean_audio = audiosamples
prev_idx = 0
for start in xrange(1, len(noisy_start)):
    prev_pos  = noisy_start[prev_idx]
    current_pos = noisy_start[start]
    diff = prev_pos - current_pos
    prev_idx += 1

    # set volume to zero for 'ahem' samples
    clean_audio[current_pos:current_pos+windowsize] = 0

# Play it back!
IPython.display.Audio(data=clean_audio, rate=44100)

# save to file and enjoy the clean episode!
librosa.output.write_wav('data/cleaned.wav', clean_audio, sr=44100)

ut.plot_waves(sound_names,raw_sounds)
ut.plot_specgram(list(sound_names[3:]), list(raw_sounds[3:]))
ut.plot_log_power_specgram(sound_names,raw_sounds)

# traditional audio features
mfccs, chroma, mel, contrast,tonnetz = ut.extract_feature('./data/jingle.wav')
ut.specgram_frombuffer(raw_sounds[0][0:44100], 6, 6, fname='./buffer.png', show=True)

# found a good model to analyze the audio features above
# and... good luck!