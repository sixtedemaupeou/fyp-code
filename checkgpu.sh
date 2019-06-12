#PBS -l nodes=gpu07
export PATH=/vol/gpudata/sd4215/miniconda3/bin:$PATH
. /vol/gpudata/cuda/9.0.176/setup.sh
export PATH=/vol/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/9.0.176/lib64:/vol/cuda/9.0.176/lib:$LD_LIBRARY_PATH
source activate mypython3
nvidia-smi
