from skimage import io
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman
import numpy as np
import random
import os

def denoise(img_dir, path):
    image = io.imread(path)
    filename = os.path.splitext(os.path.basename(path))[0]
    
    new_path_bregman = 'denoised_images/orig/bregman/' + img_dir + '/' + filename + '.png'
    io.imsave(new_path_bregman, np.clip(denoise_tv_bregman(image, weight=10), -1, 1))

    new_path_bregman = 'denoised_images/orig/chambolle/' + img_dir + '/' + filename + '.png'
    io.imsave(new_path_bregman, np.clip(denoise_tv_chambolle(image, weight=0.1, multichannel=True), -1, 1))

if __name__ == '__main__':
    images_folder = '/vol/gpudata/sd4215/segmentation/swiftnet/datasets/Cityscapes/img/left/leftImg8bit/val/'
    for img_dir in ['frankfurt', 'lindau', 'munster']:
        for img_name in os.listdir(images_folder+img_dir):
            img_path = images_folder+img_dir+'/'+img_name
            denoise(img_dir, img_path)
