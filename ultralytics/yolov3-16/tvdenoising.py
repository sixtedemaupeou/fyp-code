from skimage import io
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman
import numpy as np
import random
import os

def denoise(path):
    image = io.imread(path)
    filename = os.path.splitext(os.path.basename(path))[0]
    
    new_path_bregman = 'denoised_images/bregman/' + filename + '.jpg'
    io.imsave(new_path_bregman, denoise_tv_bregman(image, weight=10))

    new_path_bregman = 'denoised_images/chambolle/' + filename + '.jpg'
    io.imsave(new_path_bregman, denoise_tv_chambolle(image, weight=0.1, multichannel=True))

if __name__ == '__main__':
  f = open("/vol/gpudata/sd4215/ultralytics/yolov3-16/coco/5k.txt", "r")
  for line in f:
      denoise(line.rstrip())
