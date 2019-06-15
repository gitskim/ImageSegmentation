import subprocess
import numpy as np
import json
from PIL import Image

# TODO: end it with the slash
MASK_PATH = '/Users/suhyunkim/git/Dnntal/img/mask/'
# TODO: end it with the slash
ORIGINAL_PATH = '/Users/suhyunkim/git/Dnntal/img/original/'

def run_command(command, logfile=None, print_output=True, return_output=True):
    # if logfile != None:
    #     command += ' |& tee ' + logfile
    output = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        executable='/bin/bash'
    ).stdout.read()
    if print_output:
        print(output)
    if return_output:
        return str(output)


# move the pictures to a certain directory and create labels
arr_picture = np.array([])
arr_total_score = np.array([])
arr_difficulty_score = np.array([])


def do():
    counter = 0
    with open('data.json') as json_file:
        data = json.load(json_file)
        for dictionary in data:
            original_link = dictionary["Labeled Data"]

            if "Masks" in dictionary:
                counter += 1
                mask = dictionary["Masks"]
                mask_link = mask["Retro"]
                filename = f"file{counter}.jpg"
                mask_filename = f"{MASK_PATH}{filename}"
                original_filename = f"{ORIGINAL_PATH}{filename}"
                run_command(f"wget {mask_link} -O {mask_filename}")
                run_command(f"wget {original_link} -O {original_filename}")


def center_crop(uncropped_im):
    im = Image.open(uncropped_im)
    width, height = im.size

    print("before cropping: " + str(width) + ", " + str(height))

    w_after_crop = 1300
    h_after_crop = 590

    left = (width - w_after_crop) // 2
    top = (height - h_after_crop) // 2
    bottom = top + h_after_crop
    right = left + w_after_crop
    crop_rectangle = (left, top, right, bottom)

    cropped_im = im.crop(crop_rectangle)
    # plt.imshow(cropped_im, cmap='gray', vmin=0, vmax=255)
    width, height = cropped_im.size
    print("after cropping: " + str(width)  + ", " + str(height))

    # masks are RGBA where A is the transparency channel
    if cropped_im.mode in ('RGBA', 'LA'):
        cropped_im = cropped_im.convert('RGB')

    cropped_im.save(uncropped_im)

def change_filename(filelist, PATH):
    # files = run_command("ls")
    for file in filelist:
        print(file)
        last = file.rsplit('/', 1)[-1]
        last = last.replace('file', '')
        newname = f"{PATH}/{last}"
        print(newname)
        run_command(f"mv {file} {newname}")