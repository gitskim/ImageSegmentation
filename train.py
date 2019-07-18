import preprocess
import numpy as np
import glob
import os
import random
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
import cv2
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

PATH_TRAIN = '/home/deepenoughlearning/ImageSegmentation/preprocessed'
PATH_TRAIN_IMAGES = '/home/deepenoughlearning/ImageSegmentation/preprocessed/original'
PATH_TRAIN_MASKS = '/home/deepenoughlearning/ImageSegmentation/preprocessed/mask'

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def get_unet_sequential(img_rows, img_cols):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))


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

    up6 = merge(
        [Convolution2D(256, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv5)), conv4],
        mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge(
        [Convolution2D(128, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv6)), conv3],
        mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge(
        [Convolution2D(64, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv7)), conv2],
        mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge(
        [Convolution2D(32, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv8)), conv1],
        mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

model = get_unet_sequential(1040, 2000)

'''
filelist_images = glob.glob(os.path.join(PATH_TRAIN + '/original/', '*.jpg'))
filelist_masks = glob.glob(os.path.join(PATH_TRAIN + '/mask/', '*.jpg'))

filelist_images = preprocess.quicksort(filelist_images)
filelist_masks = preprocess.quicksort(filelist_masks)

print(np.array(filelist_images).shape)

train_loaded_images = []
train_loaded_masks = []

for image in filelist_images:
    img = cv2.imread(image)
    train_loaded_images.append(img)

train_loaded_images = np.array(train_loaded_images)

print("suhyun")
print(train_loaded_images.shape)

for mask in filelist_masks:
    msk = cv2.imread(mask)
    train_loaded_masks.append(msk)

print("one")
image_datagen = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   rotation_range=10
                                   )
print("two")
mask_datagen = ImageDataGenerator(horizontal_flip=True,
                                  vertical_flip=True,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  zoom_range=0.1,
                                  rotation_range=10
                                  )
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

print("three")
image_datagen.fit(train_loaded_images, augment=True, seed=seed)
print("four")
mask_datagen.fit(train_loaded_masks, augment=True, seed=seed)

print("five")

image_generator = image_datagen.flow_from_directory(
    directory=PATH_TRAIN_IMAGES
)
print("six")

mask_generator = mask_datagen.flow_from_directory(
    directory=PATH_TRAIN_MASKS
)

print("seven")

train_generator = zip(image_generator, mask_generator)

model = get_unet_sequential(1040, 2000)

print("eight")

# steps_per_epoch = number of batch iterations before a training epoch is considered finished.
model.fit_generator(
    train_generator,
    steps_per_epoch=1000,
    epochs=15,
    shuffle=True
)

print("nine")
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
    train_images = []
    train_masks = []
    val_images = []
    val_masks = []

    for j in range(folds[i][0], folds[i][1] + 1):
        # TODO: append an actual image
        val_images.append(filelist_images[j])
        val_masks.append(filelist_masks[j])

    for j in range(len(folds)):
        if j == i:
            continue

        for k in range(folds[j][0], folds[j][1] + 1):
            train_images.append(filelist_images[k])
            train_masks.append(filelist_masks[k])

    generator = gen.flow(train_images, train_masks, batch_size=batch_size)
    model = get_unet(1040, 2000)

    # steps_per_epoch = number of batch iterations before a training epoch is considered finished.
    model.fit_generator(
        generator,
        steps_per_epoch=len(filelist_images) / batch_size,
        epochs=15,
        shuffle=True,
        validation_data=(train_images, train_masks)
    )
'''
