export PATH=/vol/gpudata/sd4215/miniconda3/bin:$PATH
export PATH=/vol/gpudata/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/gpudata/cuda/9.0.176/lib64:/vol/gpudata/cuda/9.0.176/lib:$LD_LIBRARY_PATH
source activate venvswift && \
. /vol/gpudata/cuda/9.0.176/setup.sh && \
python bayesopt.py

