import preprocess
import numpy as np
import glob
import os
import random
import keras
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
import cv2
# from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Concatenate
from keras.models import *
from keras.layers import *
from keras.optimizers import *
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


# Custom IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


def get_unet(img_rows, img_cols):
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-5), loss=bce_dice_loss, metrics=[mean_iou])

    return model


def data_gen(img_folder, mask_folder, batchsize):
    num_training_images = os.listdir(img_folder)
    random.shuffle(num_training_images)  # then what happens to the mask folder?
    # it's necessary for yield
    start = 0
    while (True):
        img = np.zeros((batchsize, 1040, 2000, 1)).astype('float')
        mask = np.zeros((batchsize, 1040, 2000, 1)).astype('float')
        for i in range(start, batchsize):
            train_img = cv2.imread(img_folder + '/' + num_training_images[i], cv2.IMREAD_GRAYSCALE) / 255.
            img[i - start] = train_img
            train_mask = cv2.imread(mask_folder + '/' + num_training_images[i], cv2.IMREAD_GRAYSCALE) / 255.
            mask[i - start] = train_mask

        start += batchsize
        if (start + batchsize >= len(num_training_images)):
            start = 0
            random.shuffle(num_training_images)

        yield img, mask


filelist_images = glob.glob(os.path.join(PATH_TRAIN + '/original/cavity/', '*.jpg'))
filelist_masks = glob.glob(os.path.join(PATH_TRAIN + '/mask/cavity/', '*.jpg'))

filelist_images = preprocess.quicksort(filelist_images)
filelist_masks = preprocess.quicksort(filelist_masks)

print(np.array(filelist_images).shape)

train_loaded_images = []
train_loaded_masks = []

for image in filelist_images:
    img = cv2.imread(image, 0)  # reading grayscale images. without it, it will have 3 color channels
    newimg = np.zeros((1040, 2000, 1), dtype=int)
    newimg[:, :, 0] = img[:, :]
    train_loaded_images.append(newimg)

train_loaded_images = np.array(train_loaded_images)

print("suhyun")
print(train_loaded_images.shape)

for mask in filelist_masks:
    newimg = np.zeros((1040, 2000, 1), dtype=int)
    msk = cv2.imread(mask, 0)
    newimg[:, :, 0] = msk[:, :]
    train_loaded_masks.append(newimg)

train_loaded_masks = np.array(train_loaded_masks)

train_loaded_masks = np.array(train_loaded_masks)

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
    batch_size=1,
    directory=PATH_TRAIN_IMAGES,
    class_mode=None,
    target_size=(1040, 2000),
    color_mode='grayscale'
)
print("six")

mask_generator = mask_datagen.flow_from_directory(
    batch_size=1,
    directory=PATH_TRAIN_MASKS,
    class_mode=None,
    target_size=(1040, 2000),
    color_mode='grayscale'
)

print("seven")

train_generator = zip(image_generator, mask_generator)

model = get_unet(1040, 2000)

print("eight")

# steps_per_epoch = number of batch iterations before a training epoch is considered finished.
batch_size = 1
model.fit_generator(
    train_generator,
    validation_steps=2, steps_per_epoch=len(train_loaded_images) / (batch_size * 2), epochs=5
)

print("nine")
'''
# create an array from 0 - 195
indices = list(range(196))
random.shuffle(indices)

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
