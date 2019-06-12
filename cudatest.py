import torch

if __name__ == '__main__':
  print(torch.version.cuda)
  cuda = torch.cuda.is_available()
  print(cuda)
