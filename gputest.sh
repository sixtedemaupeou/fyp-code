export PATH=/vol/gpudata/sd4215/miniconda3/bin:$PATH
source activate mypython3
export PATH=/vol/gpudata/cuda/10.0.130/bin:$PATH
export LD_LIBRARY_PATH=/vol/gpudata/cuda/10.0.130/lib64:/vol/gpudata/cuda/10.0.130/lib:$LD_LIBRARY_PATH
. /vol/gpudata/cuda/10.0.130/setup.sh && python /vol/gpudata/sd4215/cudatest.py
