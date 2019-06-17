import preprocess
import numpy as np
import glob
import os
import random
from keras.preprocessing.image import ImageDataGenerator

PATH_TRAIN = '/Users/suhyunkim/git/Dnntal/preprocessed'

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
