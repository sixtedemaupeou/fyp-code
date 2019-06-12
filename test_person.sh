#PBS -l nodes=gpu02
export PATH=/vol/gpudata/sd4215/miniconda3/bin:$PATH

. /vol/gpudata/cuda/9.0.176/setup.sh
export PATH=/vol/cuda/9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/vol/cuda/9.0.176/lib64:/vol/cuda/9.0.176/lib:$LD_LIBRARY_PATH
source activate mypython3
# folders=("yolov3-16")
# for folder in ${folders[*]}
# do
#   for i in {1..20}
#   do
#     cp /vol/gpudata/sd4215/ultralytics/$folder/data/samples/$i/*.jpg /vol/gpudata/sd4215/ultralytics/$folder/coco/images/val2014/
#     cd /vol/gpudata/sd4215/ultralytics/$folder && python3 test.py --weights weights/yolov3.weights
#     rm /vol/gpudata/sd4215/ultralytics/$folder/coco/images/val2014/*.jpg
#   done
# done

cp /vol/gpudata/sd4215/ultralytics/yolov3-16/data/samples/perlin/perturbed/*.jpg /vol/gpudata/sd4215/ultralytics/yolov3-16/coco/images/val2014/
cd /vol/gpudata/sd4215/ultralytics/yolov3-16 && python3 test.py --weights weights/yolov3.weights
rm /vol/gpudata/sd4215/ultralytics/yolov3-16/coco/images/val2014/*.jpg
