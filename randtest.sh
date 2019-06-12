#PBS -l nodes=gpu01
export PATH=/vol/gpudata/sd4215/miniconda3/bin:$PATH

. /vol/gpudata/cuda/10.0.130/setup.sh
export PATH=/vol/cuda/10.0.130/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/10.0.130/lib64:/vol/cuda/10.0.130/lib:$LD_LIBRARY_PATH
source activate mypython3
folders=("2" "6" "12" "16")
for folder in ${folders[*]}
do
  for i in {1..20}
  do
    cp /vol/gpudata/sd4215/random/yolov3-random/data/samples/$folder/$i/*.jpg /vol/gpudata/sd4215/ultralytics/yolov3/coco/images/val2014/
    cd /vol/gpudata/sd4215/ultralytics/yolov3 && python3 test.py --weights weights/yolov3.weights
    rm /vol/gpudata/sd4215/ultralytics/yolov3/coco/images/val2014/*.jpg
  done
done
