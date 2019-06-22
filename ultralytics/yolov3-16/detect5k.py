from detect import detect

if __name__ == '__main__':
  f = open("/vol/gpudata/sd4215/ultralytics/yolov3-16/5k.txt", "r")
  for line in f:
    detect(line.rstrip())
