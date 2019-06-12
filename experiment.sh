#PBS -l nodes=gpu04
export PATH=/vol/gpudata/sd4215/miniconda3/bin:$PATH

. /vol/gpudata/cuda/10.0.130/setup.sh
export PATH=/vol/cuda/10.0.130/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/10.0.130/lib64:/vol/cuda/10.0.130/lib:$LD_LIBRARY_PATH
source activate mypython3

# mv /vol/gpudata/sd4215/ultralytics/yolov3-16/coco/images/val2014/val/*.jpg /vol/gpudata/sd4215/ultralytics/yolov3-16/coco/images/val2014/ \
cd /vol/gpudata/sd4215/ultralytics/yolov3-16 \
# && python3 tvdenoising.py \
# && cp /vol/gpudata/sd4215/ultralytics/yolov3-16/denoised_images/bergman/*.jpg /vol/gpudata/sd4215/ultralytics/yolov3-16/coco/images/val2014/ \
# && python3 test.py \
# && cp /vol/gpudata/sd4215/ultralytics/yolov3-16/denoised_images/chambolle/*.jpg /vol/gpudata/sd4215/ultralytics/yolov3-16/coco/images/val2014/ \
# && python3 test.py
# && mv /vol/gpudata/sd4215/ultralytics/yolov3-16/coco/images/val2014/*.jpg /vol/gpudata/sd4215/ultralytics/yolov3-16/coco/images/val2014/val
# && mv /vol/gpudata/sd4215/ultralytics/yolov3-16/data/samples/*.jpg /vol/gpudata/sd4215/ultralytics/yolov3-16/coco/images/val2014/ \
# && cd /vol/gpudata/sd4215/ultralytics/yolov3-16 && python3 test.py --weights weights/yolov3-16.weights \
# && rm /vol/gpudata/sd4215/ultralytics/yolov3-16/coco/images/val2014/*.jpg
&& python3 test.py --weights weights/yolov3-16.weights
