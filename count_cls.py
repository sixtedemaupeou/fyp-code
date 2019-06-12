import os

def count_classes():
  a = [0] * 80
  path = '/vol/gpudata/sd4215/ultralytics/yolov3-16/5klabels.txt'
  for file in open(path, 'r'):
    try:
      f = open(file.rstrip("\n\r"), 'r')
      for line in f:
        a[int(line.split(' ')[0])]+=1
    except IOError:
      print 'oops'
  print(a)
  return a

if __name__ == '__main__':
  count_classes()
