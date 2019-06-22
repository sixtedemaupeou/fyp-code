import os
import numpy as np
import GPyOpt
import random
from skimage import io
from noise import pnoise2
from eval import eval
import shutil

# Normalize vector
def normalize(vec):
  vmax = np.amax(vec)
  vmin  = np.amin(vec)
  return (vec - vmin) / (vmax - vmin)

# Perturb original image and clip to maximum perturbation
def perturb(orig, max_norm, noise):
  noise = np.clip(noise, np.maximum(-orig, -max_norm), np.minimum(255 - orig, max_norm))
  return (orig + noise)

#Helper function for Bayesian optimisation
# Assumes original image has shape (dim, dim, 3)
# Includes bounds for Bayesian optimization
def get_noise_f(dim_x, dim_y, channels, max_norm):
  def noise_func(params):
    freq, freq_sin, octave = params
    octave = int(octave) 

    # Base Perlin noise
    noise = np.empty((dim_x, dim_y), dtype = np.float32)
    for x in range(dim_x):
      for y in range(dim_y):
        noise[x][y] = pnoise2(x * freq, y * freq, octaves = octave)
  
    # Preprocessing and sine function color map
    noise = normalize(noise)
    noise = np.sin(noise * freq_sin * np.pi)
    noise = np.repeat(noise, channels)
    noise = noise.reshape(dim_x, dim_y, channels)
    return np.sign(noise) * max_norm
    
  # Parameter boundaries for Bayesian optimization
  bounds = [{'name' : 'freq', 'type' : 'continuous', 'domain' : (1 / 160, 1 / 20), 'dimensionality' : 1},
            {'name' : 'freq_sin', 'type' : 'continuous', 'domain' : (4, 32), 'dimensionality' : 1},
            {'name' : 'octave', 'type' : 'discrete', 'domain' : (1, 2, 3, 4), 'dimensionality' : 1}]
  return noise_func, bounds


def perlin_random(max_norm):

  images_folder = '/vol/gpudata/sd4215/segmentation/swiftnet/datasets/Cityscapes/img/left/leftImg8bit/'
  best_iou = 100

  for i in range(25):
    noise_func, bounds = get_noise_f(1024, 2048, 3, max_norm)
    freq = random.uniform(1/160, 1/20)
    freq_sin = random.uniform(4, 32)
    octave = random.randint(1, 4)
    curr_noise = noise_func((freq, freq_sin, octave))
    if os.path.isdir(images_folder+'val/'):
      shutil.rmtree(images_folder+'val/')
    shutil.copytree(images_folder+'bo-temp/val/', images_folder+'val/')
    for img_dir in ['frankfurt', 'lindau', 'munster']:
      for img_name in os.listdir(images_folder+'val/'+img_dir):
        img_path = images_folder+'val/'+img_dir+'/'+img_name
        orig_img = io.imread(img_path).astype(np.float)
        payload = perturb(orig_img, max_norm, curr_noise)
        io.imsave(fname=img_path, arr=payload.astype(np.uint8))
    iou = eval('configs/pyramid.py')

    if iou < best_iou:
      best_iou = iou
    print(best_iou)

  shutil.copytree('/vol/gpudata/sd4215/segmentation/swiftnet/configs/out/val', '/vol/gpudata/sd4215/segmentation/swiftnet/results/random/' + str(max_norm))
  return payload

if __name__ == '__main__':
  for max_norm in [6, 12]:
    perlin_random(max_norm)
