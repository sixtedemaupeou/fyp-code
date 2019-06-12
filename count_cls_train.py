import os

def count_classes():
  a = [0] * 80
  path = '/vol/gpudata/sd4215/ultralytics/yolov3-16/output_original/'
  for file in os.listdir(path):
    with open(path + file, 'r') as f:
      for line in f:
        a[int(line.split(' ')[4])]+=1
  print(a)
  return a

if __name__ == '__main__':
  count_classes()
