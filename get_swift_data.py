import re
import argparse

parser = argparse.ArgumentParser(description='Get data')
parser.add_argument('filename', type=str, help='Path to source file')

def get_data(filename):
  objects = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'IoU mean class accuracy', 'mean class recall', 'mean class precision', 'pixel accuracy']
  for obj in objects:
    with open(filename, 'r') as f:
      print(obj)
      min_obj = 100
      pattern = '^.*'+obj+'.*= ([0-9]{1,2}\.[0-9]{1,2}) %.*$'
      for line in f:
        if obj in line:
          match = re.search(pattern, line)
          min_obj = min(min_obj, float(match.group(1)))
          print(min_obj)

if __name__ == '__main__':
    args = parser.parse_args()
    get_data(args.filename)
