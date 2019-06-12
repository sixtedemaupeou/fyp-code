#PBS -l nodes=gpu01
export PATH=/vol/gpudata/sd4215/miniconda3/bin:$PATH

. /vol/gpudata/cuda/9.0.176/setup.sh
export PATH=/vol/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/9.0.176/lib64:/vol/cuda/9.0.176/lib:$LD_LIBRARY_PATH
source activate mypython3

cd /vol/gpudata/sd4215/ultralytics/yolov3-12 \
&& python3 detect.py --images /vol/gpudata/sd4215/ultralytics/yolov3-12/data/samples/20/val2014/val/COCO_val2014_000000031521.jpg
