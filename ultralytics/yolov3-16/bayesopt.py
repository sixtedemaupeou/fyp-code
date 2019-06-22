import os
import numpy as np
import GPyOpt
from skimage import io
from detect import detect
from noise import pnoise2

# Normalize vector
def normalize(vec):
  vmax = np.amax(vec)
  vmin  = np.amin(vec)
  return (vec - vmin) / (vmax - vmin)

# Perturb original image and clip to maximum perturbation
def perturb(orig, max_norm, noise):
  noise = np.sign(noise) * max_norm
  noise = np.clip(noise, np.maximum(-orig, -max_norm), np.minimum(255 - orig, max_norm))
  return (orig + noise)

#Helper function for Bayesian optimisation
# Assumes original image has shape (dim, dim, 3)
# Includes bounds for Bayesian optimization
def get_noise_f(dim_x, dim_y, channels):
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
  return sum(ious) / len(target_result)


def get_avg_iou_person(target_result, actual_result):
  # shape [[x1, y1, x2, y2, conf, cls_conf, cls]]
  if target_result == [] or actual_result == []:
    return 0
  target_result = target_result[0].detach().cpu()
  actual_result = actual_result[0].detach().cpu()
  # If no detection want to return 0
  ious = [0]
  for target_res in target_result:
    for actual_res in actual_result:
      target_cls = target_res[6]
      actual_cls = actual_res[6]
      
      if actual_cls == target_cls and target_cls == 0:
        target_x1 = target_res[0]
        target_x2 = target_res[2]
        target_y1 = target_res[1]
        target_y2 = target_res[3]
        actual_x1 = actual_res[0]
        actual_x2 = actual_res[2]
        actual_y1 = actual_res[1]
        actual_y2 = actual_res[3]
        curr_iou = get_iou_one_obj(target_x1, target_x2, target_y1, target_y2, actual_x1, actual_x2, actual_y1, actual_y2)
        if curr_iou >= 0.5:
          ious.append(curr_iou)
          break
  return sum(ious) / len(target_result)


def bayes_opt(image_path):
  print(image_path)
  max_norm = 16
  max_query = 20
  init_query = 5
  new_path = 'data/samples/20/' + os.path.splitext(os.path.basename(image_path))[0] + '.jpg'

  orig_image = io.imread(image_path).astype(np.float)
  (dim_x, dim_y, *channels) = orig_image.shape
  channels = channels[0] if channels else 1
  noise_func, bounds = get_noise_f(dim_x, dim_y, channels)
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
          return get_avg_iou_person(target_result, detect(new_path))
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
      save_path = 'data/samples/' + str(queries) + '/' + os.path.splitext(os.path.basename(image_path))[0] + '.jpg'
      io.imsave(fname=save_path, arr=payload.astype(np.uint8))
  return payload

if __name__ == '__main__':
  f = open("/vol/gpudata/sd4215/ultralytics/yolov3-16/5k.txt", "r")
  for line in f:
    bayes_opt(line.rstrip())
