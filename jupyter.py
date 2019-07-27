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
    model.compile(optimizer=Adam(lr=1e-5), loss=bce_dice_loss, metrics=[dice_coef_loss])

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

from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
checkpoint = ModelCheckpoint('unet-7-27.h5', monitor='loss', verbose=1, save_best_only=True)

model.fit_generator(
    train_generator,
    validation_steps=2, steps_per_epoch=len(train_loaded_images) / (batch_size * 2), epochs=5,
    callbacks=[early_stop, checkpoint]
)

model.save_weights("unet-7-27.h5")
model.load_weights('unet-7-27.h5')
img = np.zeros((1, 1040, 2000, 1)).astype('float')
train_img = cv2.imread(PATH_TRAIN_IMAGES + '/' + "186.jpg", cv2.IMREAD_GRAYSCALE) / 255.
img[0] = train_img

preds_train = model.predict(img, verbose=1)

mask = np.zeros((1, 1040, 2000, 1)).astype('float')
mask_img = cv2.imread(PATH_TRAIN_MASKS + '/' + "186.jpg", cv2.IMREAD_GRAYSCALE) / 255.
mask[0] = mask_img

print("nine")

plot_sample(preds_train, mask)


