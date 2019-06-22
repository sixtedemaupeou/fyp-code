import os
import numpy as np
import GPyOpt
import cv2
from skimage import io
from detect import detect
from noise import pnoise2

# Color noise
def colorize(noise, color = [1, 1, 1]):
    """
    noise           has dimension 2 or 3, pixel range [0, 1]
    color            is [a, b, c] where a, b, c are from {-1, 0, 1}
    """
    if noise.ndim == 2: # expand to include color channels
        noise = np.expand_dims(noise, 2)
    return (noise - 0.5) * color * 2 # output pixel range [-1, 1]


# Normalize variance spectrum
def normalize_var(orig):
    """
    Implementation based on https://hal.inria.fr/hal-01349134/document 
    Fabrice Neyret, Eric Heitz. Understanding and controlling contrast oscillations in stochastic texture
    algorithms using Spectrum of Variance. [Research Report] LJK / Grenoble University - INRIA. 2016,
    pp.8. <hal-01349134>
    """
    
    # Spectral variance
    mean = np.mean(orig)
    spec_var = np.fft.fft2(np.square(orig -  mean))
    
    # Normalization
    imC = np.sqrt(abs(np.real(np.fft.ifft2(spec_var))))
    imC /= np.max(imC)
    minC = 0.001
    imK =  (minC + 1) / (minC + imC)
    
    img = mean + (orig -  mean) * imK    
    return normalize(img)


# Normalize vector
def normalize(vec):
  vmax = np.amax(vec)
  vmin  = np.amin(vec)
  return (vec - vmin) / (vmax - vmin)


# Valid positions for Gabor noise
def valid_position(size_x, size_y, x, y):
    if x < 0 or x >= size_x: return False
    if y < 0 or y >= size_y: return False
    return True


### Procedural Noise ###
# Note: Do not take these as optimized implementations.

# Gabor kernel
def gaborK(ksize, sigma, theta, lambd, xy_ratio, sides):
    """
    sigma       variance of gaussian envelope
    theta         orientation
    lambd       sinusoid wavelength, bandwidth
    xy_ratio    value of x/y
    psi            phase shift of cosine in kernel
    sides        number of directions
    """
    gabor_kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, xy_ratio, 0, ktype = cv2.CV_32F)
    for i in range(1, sides):
        gabor_kern += cv2.getGaborKernel((ksize, ksize), sigma, theta + np.pi * i / sides, lambd, xy_ratio, 0, ktype = cv2.CV_32F)
    return gabor_kern

# Gabor noise - random
def gaborN_rand(size_x, size_y, grid, num_kern, ksize, sigma, theta, lambd, xy_ratio = 1, sides = 1, seed = 0):
    '''
    Gabor noise - randomly distributed kernels
    '''
    np.random.seed(seed)
    
    # Gabor kernel
    if sides != 1: gabor_kern = gaborK(ksize, sigma, theta, lambd, xy_ratio, sides)
    else: gabor_kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, xy_ratio, 0, ktype = cv2.CV_32F)
    
    # Sparse convolution noise
    sp_conv = np.zeros([size_x, size_y])
    dim_x = int(size_x / 2 // grid)
    dim_y = int(size_y / 2 // grid)
    noise = []
    for i in range(-dim_x, dim_x + 1):
        for j in range(-dim_y, dim_y + 1):
            x = i * grid + size_x / 2 - grid / 2
            y = j * grid + size_y / 2 - grid / 2
            for _ in range(num_kern):
                dx = np.random.randint(0, grid)
                dy = np.random.randint(0, grid)
                while not valid_position(size_x, size_y, x + dx, y + dy):
                    dx = np.random.randint(0, grid)
                    dy = np.random.randint(0, grid)
                weight = np.random.random() * 2 - 1
                sp_conv[int(x + dx)][int(y + dy)] = weight
    
    sp_conv = cv2.filter2D(sp_conv, -1, gabor_kern)
    return normalize(sp_conv)

# Gabor noise - uniform
def gaborN_uni(size, grid, ksize, sigma, lambd, xy_ratio, thetas):
    """
    Gabor noise - controlled, uniformly distributed kernels

    grid        ideally is odd and a factor of size
    thetas    orientation of kernels, has length (size / grid)^2
    """
    sp_conv = np.zeros([size, size])
    temp_conv = np.zeros([size, size])
    dim = int(size / 2 // grid)
    
    for i in range(-dim, dim + 1):
        for j in range(-dim, dim + 1):
            x = i * grid + size // 2
            y = j * grid + size // 2
            temp_conv[x][y] = 1
            theta = thetas[(i + dim) * dim * 2 + (j + dim)]
            
            # Gabor kernel
            gabor_kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, xy_ratio, 0, ktype = cv2.CV_32F)
            sp_conv += cv2.filter2D(temp_conv, -1, gabor_kern)
            temp_conv[x][y] = 0
    
    return normalize(sp_conv)


# Perturb original image and clip to maximum perturbation
def perturb(orig, max_norm, noise):
  noise = np.sign(noise) * max_norm
  noise = np.clip(noise, np.maximum(-orig, -max_norm), np.minimum(255 - orig, max_norm))
  return (orig + noise)

#Helper function for Bayesian optimisation
# Assumes original image has shape (dim, dim, channels)
# Includes bounds for Bayesian optimization
def get_noise_f(dim_x, dim_y, channels, noise_f):
  
    # Gabor noise - anisotropic, random
    if noise_f == 'gba':
        # if size == 224: grid = 14
        # if size == 299: grid = 13
        grid = 16
        num_kern = grid
        ksize = 23
        
        def noise_func(params):
            frac_sig, theta, frac_lambd, sides = params
            sigma = frac_sig * (ksize / 2) + 1
            lambd = frac_lambd * (ksize / 2) + 1
            noise = gaborN_rand(dim_x, dim_y, grid, num_kern, ksize, sigma, theta, lambd, sides = int(sides))
            noise = normalize_var(noise)
            if channels > 1:
              noise = colorize(noise)
            return noise
        
        bounds = [{'name' : 'kernel_var', 'type' : 'continuous', 'domain' : (0, 1), 'dimensionality' : 1},
                         {'name' : 'orientation', 'type' : 'continuous', 'domain' : (0, np.pi), 'dimensionality' : 1},
                         {'name' : 'bandwidth'  , 'type' : 'continuous'  , 'domain' : (0, 1), 'dimensionality' : 1},
                         {'name' : 'sides'  , 'type' : 'discrete'  , 'domain' : (1, 2, 3, 4, 5, 6, 7, 8), 'dimensionality' : 1}]
  
    # Perlin noise
    if noise_f == 'per':
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
          if channels > 1:
            noise = np.repeat(noise, channels)
            return noise.reshape(dim_x, dim_y, channels)
          else:
            return noise
          
        # Parameter boundaries for Bayesian optimization
        bounds = [{'name' : 'freq', 'type' : 'continuous', 'domain' : (1 / 160, 1 / 20), 'dimensionality' : 1},
                  {'name' : 'freq_sin', 'type' : 'continuous', 'domain' : (4, 32), 'dimensionality' : 1},
                  {'name' : 'octave', 'type' : 'discrete', 'domain' : (1, 2, 3, 4), 'dimensionality' : 1}]
    
    # Random noise - uniform
    if noise_f == 'ran':
        def noise_func(params):
            np.random.seed(0)
            noise = np.random.uniform(low = -1, high = 1, size = (dim_x, dim_y, channels))
            return noise
        
        # Parameter boundaries for Bayesian optimization
        bounds = []

    return noise_func, bounds

def get_iou_one_obj(x1, x2, y1, y2, x3, x4, y3, y4):
  int_x_left = max(x1, x3)
  int_x_right = min(x2, x4)
  int_y_top = max(y1, y3)
  int_y_bot = min(y2, y4)

  if int_x_left < int_x_right and int_y_top < int_y_bot:
    int_area = (int_x_right - int_x_left) * (int_y_bot - int_y_top)
    uni_area = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - int_area
    iou = int_area / uni_area
  else:
    iou = 0

  return iou


# Assumes target result always contains at least one detection
def get_avg_iou(target_result, actual_result):
  # shape [[x1, y1, x2, y2, conf, cls_conf, cls]]
  if target_result == [] or actual_result == []:
    return 0
  target_result = target_result[0].detach().cpu()
  actual_result = actual_result[0].detach().cpu()
  # If no detection want to return 0
  ious = [0]
  for target_res in target_result:
    for actual_res in actual_result:
      target_x1 = target_res[0]
      target_x2 = target_res[2]
      target_y1 = target_res[1]
      target_y2 = target_res[3]
      target_cls = target_res[6]
      
      
      actual_x1 = actual_res[0]
      actual_x2 = actual_res[2]
      actual_y1 = actual_res[1]
      actual_y2 = actual_res[3]
      actual_cls = actual_res[6]
      
      if actual_cls == target_cls:
        curr_iou = get_iou_one_obj(target_x1, target_x2, target_y1, target_y2, actual_x1, actual_x2, actual_y1, actual_y2)
        if curr_iou >= 0.5:
          ious.append(curr_iou)
          break
  return sum(ious) / len(actual_result)

def get_obj_conf(result):
  objective = 0
  if result == []:
    return 0
  for res in result[0].detach().cpu():
    # 1 is encoding of person class
    # Index 6 is class
    if res[6] == 0:
      objective += res[5] * res[4]
    # else:
    #   # Index 5 is class confidence
    #   objective -= res[5]
  return objective


def bayes_opt(image_path):
  max_norm = 16
  max_query = 20
  init_query = 5
  new_path = 'data/samples/gabor/20/' + os.path.splitext(os.path.basename(image_path))[0] + '.jpg'

  orig_image = io.imread(image_path).astype(np.float)
  (dim_x, dim_y, *channels) = orig_image.shape
  channels = channels[0] if channels else 1
  noise_func, bounds = get_noise_f(dim_x, dim_y, channels, 'gba')
  target_result = detect(image_path)
  
  # Initial queries for Bayesian optimization
  np.random.seed(0)
  feasible_space = GPyOpt.Design_space(space = bounds)        
  initial_design = GPyOpt.experiment_design.initial_design('random', feasible_space, init_query)
  
  # Objective function
  class objective_func:
      def __init__(self):
          pass
  #     # Maximises the difference in objects detected
      def f(self, params):
          params = params[0]
          payload = perturb(orig_image, max_norm, noise_func(params))
          io.imsave(fname=new_path, arr=payload.astype(np.uint8)) 
          return get_avg_iou(target_result, detect(new_path))
  queries = 0
  obj_func = objective_func()
  
  # Gaussian process and Bayesian optimization
  objective = GPyOpt.core.task.SingleObjective(obj_func.f, num_cores = 1)
  gp_model = GPyOpt.models.GPModel(exact_feval = False, optimize_restarts = 5, verbose = False)
  aquisition_opt = GPyOpt.optimization.AcquisitionOptimizer(feasible_space)
  acquisition = GPyOpt.acquisitions.AcquisitionEI(gp_model, feasible_space, optimizer = aquisition_opt)
  evaluator = GPyOpt.core.evaluators.Sequential(acquisition, batch_size = 1)

  BOpt = GPyOpt.methods.ModularBayesianOptimization(gp_model, feasible_space, objective, acquisition, evaluator, initial_design)

  while queries < max_query:
      queries += 1
      BOpt.run_optimization(max_iter = 1)
      # Evaluate best parameters
      params = BOpt.x_opt
      payload = perturb(orig_image, max_norm, noise_func(params))
      save_path = 'data/samples/gabor/' + str(queries) + '/' + os.path.splitext(os.path.basename(image_path))[0] + '.jpg'
      io.imsave(fname=save_path, arr=payload.astype(np.uint8))
  return payload

if __name__ == '__main__':
  f = open("/vol/gpudata/sd4215/ultralytics/yolov3/coco/5k.txt", "r")
  for lineno, line in enumerate(f):
      if lineno >= 1338:
        bayes_opt(line.rstrip())
