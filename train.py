import preprocess
import numpy as np
import glob
import os
import random

PATH_TRAIN = '/Users/suhyunkim/git/Dnntal/preprocessed'

# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)
#
#
# def dice_coef_loss(y_true, y_pred):
#     return 1.-dice_coef(y_true, y_pred)
#
#
# def get_unet():
#     inputs = Input((1, img_rows, img_cols))


filelist_originals = glob.glob(os.path.join(PATH_TRAIN + '/original/', '*.jpg'))
filelist_masks = glob.glob(os.path.join(PATH_TRAIN + '/mask/', '*.jpg'))

filelist_originals = preprocess.quicksort(filelist_originals)
filelist_masks = preprocess.quicksort(filelist_masks)

print(np.array(filelist_originals).shape)

# create an array from 0 - 195
indices = list(range(196))
random.shuffle(indices)

print(indices)

# TODO: make k-fold more generic
# 196 - 49 = 147
for i in range(0, 196, 49):
    print(f"i: {i}, next: {i + 49 - 1}")

folds = [[0, 48], [49, 97], [98, 146], [147, 195]]

for i in range(len(folds)):
    train_originals = []
    train_masks = []
    val_originals =[]
    val_masks = []

    for j in range(folds[i][0], folds[i][1] + 1):
        # TODO: append an actual image
        val_originals.append(j)
        val_masks.append(j)

    for j in range(len(folds)):
        if j == i:
            continue

        for k in range(folds[j][0], folds[j][1] + 1):
            train_originals.append(k)
            train_masks.append(k)


print(f"val: {len(val_originals)}")
print(f"train: {len(train_originals)}")