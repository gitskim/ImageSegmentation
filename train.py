import preprocess
import numpy as np
import glob
import os
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout

PATH_TRAIN = '/Users/suhyunkim/git/Dnntal/preprocessed'

def get_unet(img_rows, img_cols):
    inputs = Input((1, img_rows, img_cols))
    # border_mode of same means there are some padding around input or feature map, making the output feature map's size same as the input's
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

filelist_originals = glob.glob(os.path.join(PATH_TRAIN + '/original/', '*.jpg'))
filelist_masks = glob.glob(os.path.join(PATH_TRAIN + '/mask/', '*.jpg'))

filelist_originals = preprocess.quicksort(filelist_originals)
filelist_masks = preprocess.quicksort(filelist_masks)

print(np.array(filelist_originals).shape)

gen = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1,
                         rotation_range=10
                         )

# create an array from 0 - 195
indices = list(range(196))
random.shuffle(indices)

print(indices)

# TODO: make k-fold more generic
# 196 - 49 = 147
for i in range(0, 196, 49):
    print(f"i: {i}, next: {i + 49 - 1}")

folds = [[0, 48], [49, 97], [98, 146], [147, 195]]

batch_size = 49
for i in range(len(folds)):
    train_originals = []
    train_masks = []
    val_originals = []
    val_masks = []

    for j in range(folds[i][0], folds[i][1] + 1):
        # TODO: append an actual image
        val_originals.append(filelist_originals[j])
        val_masks.append(filelist_masks[j])

    for j in range(len(folds)):
        if j == i:
            continue

        for k in range(folds[j][0], folds[j][1] + 1):
            train_originals.append(filelist_originals[k])
            train_masks.append(filelist_masks[k])

    generator = gen.flow(train_originals, train_masks, batch_size=batch_size)
    model = get_unet()

    # steps_per_epoch = number of batch iterations before a training epoch is considered finished.
    model.fit_generator(
        generator,
        steps_per_epoch=len(filelist_originals) / batch_size,
        epochs=15,
        shuffle=True,
        validation_data=(train_originals, train_masks)
    )
