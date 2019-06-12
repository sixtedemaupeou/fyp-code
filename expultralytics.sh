#PBS -l gpu_mem=16GB
export PATH=/vol/gpudata/sd4215/miniconda3/bin:$PATH

. /vol/gpudata/cuda/10.0.130/setup.sh
. /vol/gpudata/miniconda3/bin
export PATH=/vol/cuda/10.0.130/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/10.0.130/lib64:/vol/cuda/10.0.130/lib:$LD_LIBRARY_PATH
source activate mypython3
cd /vol/gpudata/sd4215/ultralytics/yolov3 && python3 test.py --weights /vol/gpudata/sd4215/yolov3/weights/yolov3.weights

