import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
from PIL import Image
from tqdm import tqdm
import argparse


segmentor = SelfiSegmentation(0);

def dir_path(path_str):
    if os.path.isdir(path_str):
        return path_str
    else:
        raise NotADirectoryError(path_str)

def extract_background(src, dst):
    flag = True
    for item in tqdm(os.listdir(src)):
        src_file_path = src +item
        if os.path.isfile(src_file_path):
            img = cv2.imread(src_file_path)
            result_image = segmentor.removeBG(img, (0,0,0))
            file_name = os.path.basename(item)
            file_name = os.path.splitext(file_name)[0]
            cv2.imwrite(dst + file_name + '.png',result_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i','--src_dir', help = 'Path to source directory', type=dir_path)
    parser.add_argument('-o','--dst_dir', help = 'Path to destination directory', type=dir_path)

    arguments = parser.parse_args()

    src = arguments.src_dir
    dst = arguments.dst_dir
    src = src+'/' if src[-1]!='/' else src
    dst = dst+'/' if dst[-1]!='/' else dst
    print('Starting process...')
    extract_background(src,dst)
    print('Process finished successfully.')