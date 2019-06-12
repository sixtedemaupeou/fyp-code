#PBS -l nodes=gpu06
export PATH=/vol/gpudata/sd4215/miniconda3/bin:$PATH

. /vol/gpudata/cuda/10.0.130/setup.sh
export PATH=/vol/cuda/10.0.130/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/10.0.130/lib64:/vol/cuda/10.0.130/lib:$LD_LIBRARY_PATH
source activate mypython3
cd /vol/gpudata/sd4215/random/yolov3-random && python3 perlinrandom.py

