import os
import numpy as np
import GPyOpt
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
    noise = np.sign(noise) * max_norm
    saved_noise = (noise + 1) / 2
    io.imsave(fname='mask.png', arr=noise.astype(np.int8))
    return noise
    
  # Parameter boundaries for Bayesian optimization
  bounds = [{'name' : 'freq', 'type' : 'continuous', 'domain' : (1 / 160, 1 / 20), 'dimensionality' : 1},
            {'name' : 'freq_sin', 'type' : 'continuous', 'domain' : (4, 32), 'dimensionality' : 1},
            {'name' : 'octave', 'type' : 'discrete', 'domain' : (1, 2, 3, 4), 'dimensionality' : 1}]
  return noise_func, bounds


def bayes_opt(max_norm):
  max_query = 20
  init_query = 5

  noise_func, bounds = get_noise_f(1024, 2048, 3, max_norm)
  images_folder = '/vol/gpudata/sd4215/segmentation/swiftnet/datasets/Cityscapes/img/left/leftImg8bit/'  

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
          curr_noise = noise_func(params)
          if os.path.isdir(images_folder+'val/'):
            shutil.rmtree(images_folder+'val/')
          shutil.copytree(images_folder+'bo-temp/val/', images_folder+'val/')
          for img_dir in ['frankfurt', 'lindau', 'munster']:
            for img_name in os.listdir(images_folder+'val/'+img_dir):
              img_path = images_folder+'val/'+img_dir+'/'+img_name
              orig_img = io.imread(img_path).astype(np.float)
              payload = perturb(orig_img, max_norm, curr_noise)
              io.imsave(fname=img_path, arr=payload.astype(np.uint8)) 
          result = eval('configs/pyramid.py')
          return result
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
      print(BOpt.fx_opt)
      # Evaluate best parameters
  params = BOpt.x_opt
  fx_opt = BOpt.fx_opt
  best_opt = BOpt.Y_best
  print(params)
  print(str(best_opt))
  print(str(fx_opt))
  # shutil.copytree('/vol/gpudata/sd4215/segmentation/swiftnet/configs/out/val', '/vol/gpudata/sd4215/segmentation/swiftnet/results/bo-recall/' + str(max_norm))
  return fx_opt

if __name__ == '__main__':
    for norm in [12]:
        bayes_opt(norm)
