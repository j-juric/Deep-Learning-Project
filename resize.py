from PIL import Image
import os
from tqdm import tqdm
import argparse

"""
This script resizes all images in the source directory to the destination directory
arguments:

src_dir dst_dir width height
"""

def dir_path(path_str):
    if os.path.isdir(path_str):
        return path_str
    else:
        raise NotADirectoryError(path_str)

def center_crop(image, new_width, new_height):
    width, height = image.size
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    return image.crop((left,top,right,bottom))

def resize(src, dst, width, height):
    for item in tqdm(os.listdir(src)):
        src_file_path = src +item
        if os.path.isfile(src_file_path):
            img = Image.open(src_file_path)
            img = center_crop(img, 300, 300)
            img_resized = img.resize((width,height))
            file_name = os.path.basename(item)
            file_name = os.path.splitext(file_name)[0]
            img_resized.save(dst + file_name + '.png','PNG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_dir', help = 'Path to source directory', type=dir_path)
    parser.add_argument('--dst_dir', help = 'Path to destination directory', type=dir_path)
    parser.add_argument('--width', help= 'Set resized picture width',type=int)
    parser.add_argument('--height', help = 'Set resized picture height',type=int)

    arguments = parser.parse_args()

    src = arguments.src_dir
    dst = arguments.dst_dir
    width = arguments.width
    height = arguments.height
    src = src+'/' if src[-1]!='/' else src
    dst = dst+'/' if dst[-1]!='/' else dst
    print('Starting process...')
    resize(src,dst,width,height)
    print('Process finished successfully.')
