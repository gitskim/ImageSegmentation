import utils
import glob
import os

PATH_TRAIN = '/Users/suhyunkim/git/Dnntal/preprocessed'


def saveInDict(filelist):
    mdict = dict()
    arr = []
    for i, file in enumerate(filelist):
        key = get_key_from_file(file)
        mdict[key] = file
        arr.append(key)
    return mdict, arr


def quicksort(filelist):
    mdict, arr = saveInDict(filelist)
    quicksort_recur(arr, 0, len(filelist) - 1)
    converted = convert_to_original(arr, mdict)
    return converted


def quicksort_recur(arr, low, high):
    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort_recur(arr, low, pivot_index - 1)
        quicksort_recur(arr, pivot_index + 1, high)


def partition(arr, low, high):
    pivot = arr[high]

    i = low - 1

    # looping only up to high - 1 since high is where pivot is located
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            temp = arr[j]
            arr[j] = arr[i]
            arr[i] = temp

    temp = arr[i + 1]
    arr[i + 1] = pivot
    arr[high] = temp
    return i + 1


def convert_to_original(number, mdict):
    return_arr = []
    for i, item in enumerate(number):
        return_arr.append(mdict[item])

    return return_arr


def get_key_from_file(text):
    last = text.rsplit('/', 1)[-1]
    firstOfLast = last.rsplit('.', 1)[0]
    return int(firstOfLast)


filelist_originals = glob.glob(os.path.join(PATH_TRAIN + '/original/', '*.jpg'))
filelist_masks = glob.glob(os.path.join(PATH_TRAIN + '/mask/', '*.jpg'))

filelist_originals = quicksort(filelist_originals)
filelist_masks = quicksort(filelist_masks)


for i in range(0, len(filelist_masks)):
    utils.enlarge(filelist_originals[i], 0, 2000, 0, 1040)
    utils.enlarge(filelist_masks[i], 0, 2000, 0, 1040)
